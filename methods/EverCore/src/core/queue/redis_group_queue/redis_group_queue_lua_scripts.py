"""
Redis Grouped Queue Lua Scripts

Provides atomic operation Lua scripts to ensure queue state consistency.
"""

# Common rebalance function definition
REBALANCE_FUNCTION = """
-- rebalance partition function
local function rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
    -- Get all active owners
    local active_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
    local owner_count = #active_owners
    
    if owner_count == 0 then
        return {0, {}}
    end
    
    -- Clean up queue_list for all owners
    for _, owner_id in ipairs(active_owners) do
        local queue_list_key = queue_list_prefix .. owner_id
        redis.call('DEL', queue_list_key)
    end
    
    -- Evenly distribute partitions
    local partitions_per_owner = math.floor(total_partitions / owner_count)
    local extra_partitions = total_partitions % owner_count
    
    -- Return assignment results in flat array format for proper conversion by Redis clients
    local assigned_partitions_flat = {}
    local partition_index = 1
    
    for i, owner_id in ipairs(active_owners) do
        local queue_list_key = queue_list_prefix .. owner_id
        local partitions_for_this_owner = partitions_per_owner
        
        -- First 'extra_partitions' owners get one additional partition
        if i <= extra_partitions then
            partitions_for_this_owner = partitions_for_this_owner + 1
        end
        
        local owner_partitions = {}
        for j = 1, partitions_for_this_owner do
            local partition_name = string.format("%03d", partition_index)
            redis.call('LPUSH', queue_list_key, partition_name)
            table.insert(owner_partitions, partition_name)
            partition_index = partition_index + 1
        end
        
        -- Set expiration time
        redis.call('EXPIRE', queue_list_key, owner_expire)
        
        -- Add owner_id and partition list to flat array
        table.insert(assigned_partitions_flat, owner_id)
        table.insert(assigned_partitions_flat, owner_partitions)
    end
    
    return {owner_count, assigned_partitions_flat}
end
"""

# Lua script for adding message to queue
ENQUEUE_SCRIPT = """
-- Parameters:
-- KEYS[1]: queue key (zset)
-- KEYS[2]: total counter key
-- ARGV[1]: message content (supports JSON string or BSON binary data)
-- ARGV[2]: sort score
-- ARGV[3]: queue expiration time (seconds)
-- ARGV[4]: activity expiration time (seconds, 7 days)
-- ARGV[5]: maximum total limit

local queue_key = KEYS[1]
local counter_key = KEYS[2]
local message = ARGV[1]
local score = tonumber(ARGV[2])
local queue_expire = tonumber(ARGV[3])
local activity_expire = tonumber(ARGV[4])
local max_total = tonumber(ARGV[5])

-- Check total limit
local current_count = tonumber(redis.call('GET', counter_key) or '0')
if current_count >= max_total then
    return {0, current_count, "Exceeded maximum total limit"}
end

-- Add message to queue
local added = redis.call('ZADD', queue_key, score, message)
if added == 1 then
    -- Update queue expiration time
    redis.call('EXPIRE', queue_key, queue_expire)
    
    -- Increment total counter
    local new_count = redis.call('INCR', counter_key)
    redis.call('EXPIRE', counter_key, activity_expire)
    
    return {1, new_count, "Added successfully"}
else
    return {0, current_count, "Message already exists"}
end
"""

# Lua script for rebalance repartitioning
REBALANCE_PARTITIONS_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- ARGV[1]: total number of partitions
-- ARGV[2]: owner expiration time (seconds, default 1 hour)

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local total_partitions = tonumber(ARGV[1])
local owner_expire = tonumber(ARGV[2])

-- Call rebalance function
return rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
"""

# Lua script for joining consumer
JOIN_CONSUMER_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- ARGV[1]: owner_id
-- ARGV[2]: current timestamp
-- ARGV[3]: owner expiration time (seconds, default 1 hour)
-- ARGV[4]: total number of partitions

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local owner_id = ARGV[1]
local current_time = tonumber(ARGV[2])
local owner_expire = tonumber(ARGV[3])
local total_partitions = tonumber(ARGV[4])

-- Join owner_activate_time_zset
redis.call('ZADD', owner_zset_key, current_time, owner_id)
redis.call('EXPIRE', owner_zset_key, owner_expire)

-- Call rebalance function
return rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
"""

# Lua script for consumer exit
EXIT_CONSUMER_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- ARGV[1]: owner_id
-- ARGV[2]: owner expiration time (seconds, default 1 hour)
-- ARGV[3]: total number of partitions

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local owner_id = ARGV[1]
local owner_expire = tonumber(ARGV[2])
local total_partitions = tonumber(ARGV[3])

-- Remove from owner_activate_time_zset
redis.call('ZREM', owner_zset_key, owner_id)

-- Delete corresponding queue_list
local queue_list_key = queue_list_prefix .. owner_id
redis.call('DEL', queue_list_key)

-- Check if there are remaining owners, if so call rebalance function
local remaining_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
if #remaining_owners == 0 then
    return {0, {}}
end

-- Call rebalance function
return rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire)
"""

# Lua script for consumer keepalive
KEEPALIVE_CONSUMER_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- ARGV[1]: owner_id
-- ARGV[2]: current timestamp
-- ARGV[3]: owner expiration time (seconds, default 1 hour)

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local owner_id = ARGV[1]
local current_time = tonumber(ARGV[2])
local owner_expire = tonumber(ARGV[3])

-- Update time in owner_activate_time_zset
local updated = redis.call('ZADD', owner_zset_key, current_time, owner_id)
redis.call('EXPIRE', owner_zset_key, owner_expire)

-- Renew expiration for corresponding queue_list
local queue_list_key = queue_list_prefix .. owner_id
local exists = redis.call('EXISTS', queue_list_key)
if exists == 1 then
    redis.call('EXPIRE', queue_list_key, owner_expire)
    return 1
else
    return 0
end
"""

# Lua script for periodic cleanup of inactive owners
CLEANUP_INACTIVE_OWNERS_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- KEYS[3]: queue_prefix (used to construct partition queue key)
-- KEYS[4]: counter_key (message total counter key)
-- ARGV[1]: inactive threshold timestamp (5 minutes ago)
-- ARGV[2]: current timestamp
-- ARGV[3]: owner expiration time (seconds, default 1 hour)
-- ARGV[4]: total number of partitions

__REBALANCE_FUNCTION__

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local queue_prefix = KEYS[3]
local counter_key = KEYS[4]
local inactive_threshold = tonumber(ARGV[1])
local current_time = tonumber(ARGV[2])
local owner_expire = tonumber(ARGV[3])
local total_partitions = tonumber(ARGV[4])

-- Get all inactive owners
local inactive_owners = redis.call('ZRANGEBYSCORE', owner_zset_key, 0, inactive_threshold)
local cleaned_count = 0

-- Clean up inactive owners
for _, owner_id in ipairs(inactive_owners) do
    -- Remove from zset
    redis.call('ZREM', owner_zset_key, owner_id)
    
    -- Delete corresponding queue_list
    local queue_list_key = queue_list_prefix .. owner_id
    redis.call('DEL', queue_list_key)
    
    cleaned_count = cleaned_count + 1
end

-- Recalculate counter_key regardless to ensure data consistency
local total_messages = 0
for i = 1, total_partitions do
    local partition_name = string.format("%03d", i)
    local queue_key = queue_prefix .. partition_name
    local queue_size = redis.call('ZCARD', queue_key)
    total_messages = total_messages + queue_size
end
redis.call('SET', counter_key, total_messages)

-- Rebalance if any cleanup occurred
local need_rebalance = cleaned_count > 0
if not need_rebalance then
    return {cleaned_count, 0, {}}
end

-- Check if there are remaining owners
local remaining_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
if #remaining_owners == 0 then
    return {cleaned_count, 0, {}}
end

-- Call rebalance function
local owner_count, assigned_partitions = unpack(rebalance_partitions(owner_zset_key, queue_list_prefix, total_partitions, owner_expire))
return {cleaned_count, owner_count, assigned_partitions}
"""

# Lua script for forced cleanup and reset (supports optional purge)
FORCE_CLEANUP_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- KEYS[3]: queue_prefix (used to construct partition queue key)
-- KEYS[4]: counter_key (message total counter key)
-- ARGV[1]: total number of partitions
-- ARGV[2]: purge_all flag ("1" to empty all partition queues and set counter to 0; otherwise only recalculate counter)

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local queue_prefix = KEYS[3]
local counter_key = KEYS[4]
local total_partitions = tonumber(ARGV[1])
local purge_all = ARGV[2]

-- Get all owners
local all_owners = redis.call('ZRANGE', owner_zset_key, 0, -1)
local cleaned_count = 0

-- Delete queue_list for all owners
for _, owner_id in ipairs(all_owners) do
    local queue_list_key = queue_list_prefix .. owner_id
    redis.call('DEL', queue_list_key)
    cleaned_count = cleaned_count + 1
end

-- Delete owner_activate_time_zset
redis.call('DEL', owner_zset_key)

if purge_all == '1' then
    -- Empty all partition queues and set counter to 0
    for i = 1, total_partitions do
        local partition_name = string.format("%03d", i)
        local queue_key = queue_prefix .. partition_name
        redis.call('DEL', queue_key)
    end
    redis.call('SET', counter_key, 0)
    return total_partitions
else
    -- Only recalculate counter
    local total_messages = 0
    for i = 1, total_partitions do
        local partition_name = string.format("%03d", i)
        local queue_key = queue_prefix .. partition_name
        local queue_size = redis.call('ZCARD', queue_key)
        total_messages = total_messages + queue_size
    end
    redis.call('SET', counter_key, total_messages)
    return cleaned_count
end
"""

# Lua script for getting messages (traverse all partitions and attempt to get one from each)
GET_MESSAGES_SCRIPT = """
-- Parameters:
-- KEYS[1]: owner_activate_time_zset key
-- KEYS[2]: queue_list_prefix (used to construct queue_list key for each owner)
-- KEYS[3]: queue_prefix (used to construct partition queue key)
-- KEYS[4]: counter_key (message total counter key)
-- ARGV[1]: owner_id
-- ARGV[2]: owner expiration time (seconds, default 1 hour)
-- ARGV[3]: score difference threshold (milliseconds)
-- ARGV[4]: current score (used for threshold comparison when queue is empty)

local owner_zset_key = KEYS[1]
local queue_list_prefix = KEYS[2]
local queue_prefix = KEYS[3]
local counter_key = KEYS[4]
local owner_id = ARGV[1]
local owner_expire = tonumber(ARGV[2])
local score_threshold = tonumber(ARGV[3])
local current_score = tonumber(ARGV[4])

-- Check if owner exists in zset
local owner_score = redis.call('ZSCORE', owner_zset_key, owner_id)
if not owner_score then
    -- Owner does not exist, need to join consumer
    return {"JOIN_REQUIRED", {}}
end

-- Check if queue_list exists
local queue_list_key = queue_list_prefix .. owner_id
local queue_list_exists = redis.call('EXISTS', queue_list_key)
if queue_list_exists == 0 then
    -- queue_list does not exist, need to join consumer
    return {"JOIN_REQUIRED", {}}
end

-- Get owner's queue list
local owner_queues = redis.call('LRANGE', queue_list_key, 0, -1)
if #owner_queues == 0 then
    return {"NO_QUEUES", {}}
end

local messages = {}
local messages_consumed = 0

-- Traverse all partitions, attempt to get 1 message from each
for _, partition in ipairs(owner_queues) do
    local queue_key = queue_prefix .. partition
    
    -- Check if queue has messages
    local queue_size = redis.call('ZCARD', queue_key)
    if queue_size > 0 then
        -- Get score of earliest message
        local min_result = redis.call('ZRANGE', queue_key, 0, 0, 'WITHSCORES')
        
        -- Directly compare difference between earliest message and current score
        if #min_result >= 2 then
            local earliest_message_score = tonumber(min_result[2])
            -- Check difference between earliest message score and current score
            if (current_score - earliest_message_score) >= score_threshold then
                -- Get earliest message (directly remove)
                local popped = redis.call('ZPOPMIN', queue_key)
                if #popped >= 2 then
                    table.insert(messages, popped[1])  -- Return only message content
                    messages_consumed = messages_consumed + 1
                end
            end
        end
    end
end

-- If messages were consumed, reduce counter_key count
if messages_consumed > 0 then
    local new_count = redis.call('DECRBY', counter_key, messages_consumed)
    -- Ensure count does not become negative
    if new_count < 0 then
        redis.call('SET', counter_key, 0)
    end
end

-- Renew queue_list expiration
redis.call('EXPIRE', queue_list_key, owner_expire)

return {"SUCCESS", messages}
"""

# Lua script for getting queue statistics
GET_QUEUE_STATS_SCRIPT = """
-- Parameters:
-- KEYS[1]: queue key (zset)
-- KEYS[2]: total counter key

local queue_key = KEYS[1]
local counter_key = KEYS[2]

-- Get queue size
local queue_size = redis.call('ZCARD', queue_key)

-- Get total count
local total_count = tonumber(redis.call('GET', counter_key) or '0')

-- Get score range of queue
local min_max = {}
if queue_size > 0 then
    local min_result = redis.call('ZRANGE', queue_key, 0, 0, 'WITHSCORES')
    local max_result = redis.call('ZRANGE', queue_key, -1, -1, 'WITHSCORES')
    if #min_result >= 2 then
        min_max.min_score = tonumber(min_result[2])
    end
    if #max_result >= 2 then
        min_max.max_score = tonumber(max_result[2])
    end
end

return {
    queue_size,
    total_count,
    min_max.min_score or 0,
    min_max.max_score or 0
}
"""

# Lua script for batch getting statistics of all partitions
GET_ALL_PARTITIONS_STATS_SCRIPT = """
-- Parameters:
-- KEYS[1]: queue_prefix (used to construct partition queue key)
-- KEYS[2]: total counter key
-- ARGV[1]: total number of partitions

local queue_prefix = KEYS[1]
local counter_key = KEYS[2]
local total_partitions = tonumber(ARGV[1])

-- Get total count
local total_count = tonumber(redis.call('GET', counter_key) or '0')

-- Store statistics for all partitions
local partition_stats = {}
local total_messages_in_queues = 0
local global_min_score = nil
local global_max_score = nil

-- Traverse all partitions
for i = 1, total_partitions do
    local partition_name = string.format("%03d", i)
    local queue_key = queue_prefix .. partition_name
    
    -- Get queue size
    local queue_size = redis.call('ZCARD', queue_key)
    total_messages_in_queues = total_messages_in_queues + queue_size
    
    local min_score = 0
    local max_score = 0
    
    if queue_size > 0 then
        -- Get minimum and maximum score
        local min_result = redis.call('ZRANGE', queue_key, 0, 0, 'WITHSCORES')
        local max_result = redis.call('ZRANGE', queue_key, -1, -1, 'WITHSCORES')
        
        if #min_result >= 2 then
            min_score = tonumber(min_result[2])
            if global_min_score == nil or min_score < global_min_score then
                global_min_score = min_score
            end
        end
        
        if #max_result >= 2 then
            max_score = tonumber(max_result[2])
            if global_max_score == nil or max_score > global_max_score then
                global_max_score = max_score
            end
        end
    end
    
    -- Store partition statistics (flat array format)
    table.insert(partition_stats, partition_name)
    table.insert(partition_stats, queue_size)
    table.insert(partition_stats, min_score)
    table.insert(partition_stats, max_score)
end

return {
    total_count,
    total_messages_in_queues,
    global_min_score or 0,
    global_max_score or 0,
    partition_stats
}
"""

# Replace __REBALANCE_FUNCTION__ placeholder when module loads
REBALANCE_PARTITIONS_SCRIPT = REBALANCE_PARTITIONS_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
JOIN_CONSUMER_SCRIPT = JOIN_CONSUMER_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
EXIT_CONSUMER_SCRIPT = EXIT_CONSUMER_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
CLEANUP_INACTIVE_OWNERS_SCRIPT = CLEANUP_INACTIVE_OWNERS_SCRIPT.replace(
    '__REBALANCE_FUNCTION__', REBALANCE_FUNCTION
)
