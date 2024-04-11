# 导入必要库
import os, re, logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

# Constant
STARTINGTIME = datetime.now().strftime("%Y%m%d%H%M%S")
CURRENTFILEPATH = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(CURRENTFILEPATH, "DATA", STARTINGTIME)
TEXTPATTERN = re.compile(r'^(\d{4}\d{2}\d{2})//(.+)$')
INPUTTEXTPATH = "./text.txt"
ERRORTEXTPATH = "errorText.txt"
TEXTDATACSV = "textData.csv"

# Bert Model Constant
MODEL_NAME = "bert-base-multilingual-cased"
BERT_BASE_MULTILINGUAL_TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
BERT_BASE_MULTILINGUAL_MODEL = TFAutoModel.from_pretrained(MODEL_NAME)
NUM_GRU_UNITS = 128

# 数据处理类，数据的导入，清洗和预处理
class textPreprocessor():
    def __init__(self, inputTextPath: str, tokenizer: AutoTokenizer, errorTextPath: str=ERRORTEXTPATH, textDataCSV: str=TEXTDATACSV) -> None:
        '''
        尽管目前很多量并没有存为属性的必要，但为了扩展性考虑，这里采取了全部存为属性的做法。
        '''
        self.textPath = os.path.abspath(inputTextPath)
        self.errorTextPath = os.path.join(DATADIR, errorTextPath)
        self.textDataCSV = os.path.join(DATADIR, textDataCSV)
        self.inputTextList = []
        self.textWashed = []
        self.textDataWashed = {}
        self.bertBaseMultilingualInputs = {}
        self.loadText()
        self.textWash()
        self.textPreprocess(tokenizer)

    def loadText(self) -> None:
        '''
        这里认为文本文件中的每一行对应一段模型要输入的文本。
        暂时假设不存在一段文本内出现“串行”的问题。
        '''
        with open(self.textPath, 'r', encoding="utf-8") as inputFile:
            self.inputTextList  = inputFile.readlines()

    def textWash(self) -> None:
        '''
        这里假设只要符合“日期//文本内容”分割方法的数据都算作正常数据。
        此外，日期需要满足 r'^\d{4}\d{2}\d{2}$' 的模式匹配，即四位年份，两位月份，两位日期。
        更加复杂的日期匹配可能需要一个专门日期类来实现（例如年份要在合理范围内，不同月份对应的日期范围不同等）。
        '''
        with open(self.errorTextPath, 'w', encoding="utf-8") as errorFile:
            # 这里假设文本列表并不会大到给内存带来显著压力
            temp_list = []
            for text in self.inputTextList:
                if TEXTPATTERN.match(text):
                    temp_list.append(text)
                else:
                    errorFile.write(text + '\n')
            self.inputTextList = temp_list

    def textPreprocess(self, tokenizer: AutoTokenizer) -> None:
        '''
        预处理以获得模型要输入的 token，并获取一个关于格式正确数据的表格
        '''
        self.bertBaseMultilingualInputs = tokenizer(
            self.inputTextList, return_tensors="tf", padding=True, truncation=True
        )
        textDF = pd.Series(self.inputTextList).str.split("//", expand=True)
        textDF.columns = ["Date", "Content"]
        textDF.to_csv(self.textDataCSV, index=False)

# Build Model Class
class simpleModel(tf.keras.Model):
    def __init__(self, bert_model, num_gru_units) -> None:
        super().__init__()
        self.bert = bert_model
        self.gruLayer0 = tf.keras.layers.GRU(num_gru_units, return_sequences=True)
        self.gruLayer1 = tf.keras.layers.GRU(num_gru_units, return_sequences=True)
        self.sumPoolingLayer0 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))
        self.sumPoolingLayer1 = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))
        self.bertOut = None
        self.bertSequenceOutput = None
        self.bertPooledOutput = None
        self.gruOutput0 = None
        self.gruOutput1 = None
        self.sumPoolingOut0 = None
        self.sumPoolingOut1 = None

    def call(self, inputTextsToken: tf.Tensor) -> tf.Tensor:
        self.bert_out = self.bert(inputTextsToken)                                      # [batch_size, sequence_length]
        self.bert_sequence_output = self.bert_out.last_hidden_state                     # [batch_size, sequence_length, bert_hidden_size]
        self.bert_pooled_output = self.bert_out.pooler_output                           # [batch_size, bert_hidden_size]
        self.gruOutput0 = self.gruLayer0(self.bert_sequence_output)                     # [batch_size, sequence_length, num_gru_units]
        self.sumPoolingOut0 = self.sumPoolingLayer0(self.gruOutput0)                    # [batch_size, num_gru_units]
        self.gruOutput1 = self.gruLayer1(tf.expand_dims(self.sumPoolingOut0, axis=1))   # [batch_size, 1, num_gru_units]
        self.sumPoolingOut1 = self.sumPoolingLayer1(self.gruOutput1)                    # [batch_size, num_gru_units]
        return(self.sumPoolingOut1)

if __name__ == "__main__":
    if not os.path.exists(DATADIR):
        os.makedirs(DATADIR)
    logging.basicConfig(filename=os.path.join(CURRENTFILEPATH, "error.log"),
                        level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')
    try:
        textData = textPreprocessor(INPUTTEXTPATH, BERT_BASE_MULTILINGUAL_TOKENIZER)
        model = simpleModel(BERT_BASE_MULTILINGUAL_MODEL, NUM_GRU_UNITS)
        outputVector = model(textData.bertBaseMultilingualInputs["input_ids"])
        np.savetxt(os.path.join(DATADIR, "output_vector.txt"), outputVector.numpy())
    except Exception as e:
        logging.error("Error occurred", exc_info=True)