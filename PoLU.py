import tensorflow as tf
from tensorflow import name_scope
import numpy as np


def polu_activate(x, n):
    r = tf.maximum(x, tf.constant(0.))
    l = tf.subtract(tf.pow((tf.subtract(1.,x)), -1 * n), 1.)
    cond = tf.less_equal(x, tf.constant(0.))
    return tf.where(cond, l, r)

def polu_deriv(x, n):
    r = tf.constant(1.)
    l = tf.multiply(n, tf.pow(tf.subtract(1., x), tf.multiply(tf.add(n, 1.), -1.)))
    cond = tf.less_equal(x, tf.constant(0.))
    return tf.where(cond, l, r)

def _PoLU(x, n, name=None):
    with tf.name_scope(name, "PoLU", [x,n]) as scope:
        