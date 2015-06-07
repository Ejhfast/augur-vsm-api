import fileinput
from collections import defaultdict
from numpy import zeros
from numpy import add
from numpy import dot
from numpy import matrix
from numpy.linalg import norm
import math
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
from flask import Flask, render_template
import flask
from bson.json_util import dumps
from bson.objectid import ObjectId
from flask_cors import *
import json

app = Flask(__name__)

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

vecs = np.load("act-vecs.npy")
noun_vecs = np.load("noun-vecs.npy")
nouns_d = np.load("nouns_d.pkl")
vps_d = np.load("acts_d.pkl")
vps = {i:k for k,i in vps_d.iteritems()}

app.config['CORS_ORIGINS'] = ['*']
app.config['CORS_HEADERS'] = ['Content-Type']


def query_vec(acts):
  search = zeros(vecs.shape[1])
  for q in acts:
    if(q in nouns_d):
      search = add(search,normalize(noun_vecs[nouns_d[q]]))
    elif(q in vps_d):
      search = add(search,normalize(vecs[vps_d[q]]))
    else:
      print("{} not in index".format(q))
  return search

def lookup(acts,n):
  search = query_vec(acts)
  results = cosine_similarity(search, vecs)
  get_names = [(vps[i],x) for i,x in enumerate(results[0]) if x > 0]
  return sorted(get_names,key=lambda x: x[1], reverse=True)[:n]

@app.route("/predict/<acts>")
@cross_origin(headers=['Content-Type'])
def predict(acts):
  acts = acts.split(",") # sketch
  results = lookup(acts,100)
  return json.dumps(results)

if __name__ == "__main__":
    app.run(debug = True,port=80)
