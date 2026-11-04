"""Write path: receive message → call everos.algo algorithms → write store.

Owner inference (``Episode.owner_id``) is owned by the algo layer
(``everos.algo``) — everos consumes algo's output and routes md path /
memcell rows by it, no inference logic here.
"""
