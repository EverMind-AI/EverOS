"""Write path: receive message → call everalgo algorithms → write store.

Owner inference (``Episode.owner_id``) is owned by the algo layer
(``everalgo``) — everos consumes algo's output and routes md path /
memcell rows by it, no inference logic here.
"""
