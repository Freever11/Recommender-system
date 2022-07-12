import tensorflow as tf
import config
#配置参数文件
FLAGS = config.FLAGS


def build_user_model(features, mode, params):

  """
  user tower embedding
  features: user_features
  mode:
  params:
  
  return: user_net
  """
  
    # 特征输入
    user_vector_net = []
    for key, value in params["feature_configs"].user_columns.items():
        user_vector_net.append(tf.feature_column.input_layer(features, value))  
        # Returns a dense Tensor as input layer based on given feature_columns.
    
    # 特征拼接
    net = tf.concat(user_vector_net, axis=1) # 在第一个维度进行拼接

    # 全连接
    for idx, units in enumerate(params["hidden_units"]):
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu, name="user_fc_layer_%s"%idx)
        net = tf.layers.dropout(net, 0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最后输出
    net = tf.layers.dense(net, units=64, name="user_output_layer")
    return net

def build_item_model(features, mode, params):

  """
  item tower embedding
  features: item_features
  mode:
  params:
  
  return: item_net
  """
  
    # 特征输入
    item_vector_net = []
    for key, value in params["feature_configs"].items_columns.items():
        item_vector_net.append(tf.feature_column.input_layer(features, value))  
        # Returns a dense Tensor as input layer based on given feature_columns.
    
    # 特征拼接
    net = tf.concat(item_vector_net, axis=1)  # 在第一个维度进行拼接

    # 全连接
    for idx, units in enumerate(params["hidden_units"]):
        net = tf.layers.dense(net, units=units, activation=tf.nn.leaky_relu, name="item_fc_layer_%s"%idx)
        net = tf.layers.dropout(net, 0.4, training=(mode == tf.estimator.ModeKeys.TRAIN))
    # 最后输出
    net = tf.layers.dense(net, units=64, name="item_output_layer")
    return net

def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = dict()
        if FLAGS.export_user_model:
            user_net = build_user_model(features, mode, params)
            predictions = {"user_vector": user_net}
        elif FLAGS.export_item_model:
            item_net = build_user_model(features, mode, params)
            predictions = {"item_vector": item_net}
        export_outputs = {"prediction": tf.estimator.export.PredictOutput(output=predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_output=export_outputs)

    user_net = build_user_model(features, mode, params)
    item_net = build_item_model(features, mode, params)

    dot = tf.reduce_sum(tf.multiply(user_net, item_net), axis=1, keep_dims=True)
    pred = tf.sigmoid(dot)

    if mode == tf.estimator.ModeKeys.EVAL:
        labels = tf.cast(labels, tf.float32)
        loss = tf.losses.log_loss(labels, pred)
        metrics = {"auc": tf.metrics.auc(labels=labels, predictions=pred)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        labels = tf.cast(labels, tf.float32)
        loss = tf.losses.log_loss(labels, pred)
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss, global_step = global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
        
##### tf.estimator.EstimatorSpec: Ops and objects returned from a model_fn and passed to an Estimator
##### EstimatorSpec fully defines the model to be run by an Estimator.
