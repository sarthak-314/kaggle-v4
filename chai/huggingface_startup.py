import tensorflow as tf

import transformers 
import datasets 
from transformers import (
    AutoTokenizer, TFAutoModel, TFAutoModel, TFAutoModelForQuestionAnswering, 
    EvalPrediction, 
)
from datasets import (
    concatenate_datasets, list_datasets, 
)

from chai.huggingface_qa_utils import *
from chai.data.qa_datasets import *
from termcolor import colored

class ChaiQAModelA(TFAutoModelForQuestionAnswering): 
    def compute_loss(self, labels, logits):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        start_logits, end_logits = logits
        
        start_loss = loss_fn(labels['start_position'], start_logits)
        end_loss = loss_fn(labels['end_position'], end_logits)

        loss = (self.start_weight * start_loss + end_loss) / (self.start_weight + 1)

        # Multiply loss by negative weight where end position is 0
        loss = tf.where(tf.squeeze(labels['end_position']) == 0, loss*self.negative_weight, loss)

        # Debugging when eager execution
        # print('labels: ', labels)
        # print('logits: ', logits)
        # print('loss: ', loss)

        return loss