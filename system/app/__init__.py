from keras import backend as K
from flask import Flask


app = Flask(__name__)

from app import view
