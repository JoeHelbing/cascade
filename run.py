import logging as log
import os

cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd, "./log/")):
    os.makedirs(os.path.join(cwd, "./log/"))

# set up logging for the model
log.basicConfig(
    filename="./log/resistance_cascade.log",
    filemode="w",
    format="%(message)s",
    level=log.DEBUG,
)

from resistance_cascade.server import server

server.launch()
