# EverOS OpenClaw Plugin

通过自然语言对话为 **OpenClaw / 龙虾** 提供持久记忆能力。

这个插件保留当前 OpenClaw 的 `context-engine` 架构，并连接到自托管的 EverOS backend。其后端能力由 [EverOS](https://github.com/EverMind-AI/EverOS) 提供。

## 它能做什么

- 在每次回复前通过 `assemble()` 自动回忆相关记忆
- 在每轮对话后通过 `afterTurn()` 自动保存新内容
- 用户只需要正常聊天
- 不需要手动调用 `memory_store` 或 `memory_search`

重要说明：

- 这是一个 `context-engine` 插件
- 它不是 `memory` slot 插件
- 为避免冲突，安装时会把 `plugins.slots.memory` 设置为 `none`

## 快速开始

推荐安装方式：

```bash
npx --yes --package @evermind-ai/openclaw-plugin everos-install
```

安装器会：

- 复用已有的 `~/.openclaw/openclaw.json`
- 把插件路径写入 `plugins.load.paths`
- 把 `evermind-ai-everos` 写入 `plugins.allow`
- 设置 `plugins.slots.contextEngine = "evermind-ai-everos"`
- 设置 `plugins.slots.memory = "none"`
- 为插件创建或补齐默认配置

安装完成后：

```bash
openclaw gateway restart
```

然后用自然语言验证：

```text
记住：我喜欢意式浓缩。
我喜欢什么咖啡？
```

## 后端

默认后端地址：

```text
http://localhost:1995
```

健康检查：

```bash
curl http://localhost:1995/health
```

如果你还没有启动 EverOS backend：

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
docker compose up -d
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
cp env.template .env
# 编辑 .env
uv run python src/run.py
```

## 自然语言记忆是如何工作的

运行时流程：

1. 用户发送一条普通消息。
2. `assemble()` 去 EverOS backend 搜索相关记忆。
3. 命中的记忆被注入为上下文。
4. OpenClaw 正常回复。
5. `afterTurn()` 把这一轮的新内容写回 EverOS backend。

所以用户看到的体验是：

- “记住：我偏好深色模式”
- 之后再问：“我偏好什么 UI 风格？”

整个过程不需要显式调用记忆工具。

## OpenClaw 配置示例

期望的配置结构如下：

```json
{
  "plugins": {
    "allow": ["evermind-ai-everos"],
    "slots": {
      "memory": "none",
      "contextEngine": "evermind-ai-everos"
    },
    "entries": {
      "evermind-ai-everos": {
        "enabled": true,
        "config": {
          "baseUrl": "http://localhost:1995",
          "userId": "everos-user",
          "groupId": "everos-group",
          "topK": 5,
          "memoryTypes": ["episodic_memory"],
          "retrieveMethod": "hybrid"
        }
      }
    }
  }
}
```

## 配置项


| 字段               | 默认值                                                           | 说明                |
| ---------------- | ------------------------------------------------------------- | ----------------- |
| `baseUrl`        | `http://localhost:1995`                                       | EverOS backend 地址 |
| `userId`         | `everos-user`                                                 | 记忆归属的用户标识         |
| `groupId`        | `everos-group`                                                | 共享记忆命名空间          |
| `topK`           | `5`                                                           | 最多检索条目数           |
| `memoryTypes`    | `["episodic_memory"]` | 要搜索的记忆类型          |
| `retrieveMethod` | `hybrid`                                                      | 检索模式              |


## 手动安装

```bash
npm install -g @evermind-ai/openclaw-plugin
everos-install
```

## 故障排查


| 问题        | 解决方式                                                                  |
| --------- | --------------------------------------------------------------------- |
| 插件未加载     | 检查 `plugins.allow`、`plugins.load.paths`、`plugins.slots.contextEngine` |
| 后端连接失败    | 检查 `baseUrl`，并执行 `curl <baseUrl>/health`                              |
| 没有回忆出记忆   | 检查后端数据，并尝试更具体的问题                                                      |
| 没有保存记忆    | 检查 EverOS backend 写入接口是否正常                                            |
| 与其他记忆插件冲突 | 确认 `plugins.slots.memory = "none"`                                    |


## 相关文件

- `index.js`：插件入口（注册）
- `src/engine.js`：ContextEngine 生命周期实现
- `src/convert.js`：OpenClaw 消息转 EverOS 格式
- `src/api.js`：EverOS backend REST API 客户端
- `src/messages.js`：消息归一化与轮次收集
- `src/prompt.js`：记忆搜索结果解析与 prompt 构建
- `src/subagent-assembler.js`：记忆上下文组装
- `bin/install.js`：安装器与配置引导
- `openclaw.plugin.json`：插件元数据与配置结构

## 许可证

Apache-2.0