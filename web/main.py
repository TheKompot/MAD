from flask import Flask, render_template, request
import sqlite3
import pandas as pd
class NaiveBayes:
    def __init__(self, data:pd.DataFrame=None):
        self.prob_tables = {}
        if data is None:
                return
        self.data = data.copy()
        for col in data.columns:
            self.prob_tables[col] = data[col].value_counts() / len(data[col])
            for index,val in self.prob_tables[col].items():
                if val == 0:
                    self.prob_tables[col][index] = 1e-6

    def predict(self, evidence:dict, target:str):
        new_probs = {}
        for val in self.data[target].unique():
            new_probs[val] = self.prob_tables[target][val]

            for ev in evidence:
                new_probs[val] *= ((self.data.query(f"{ev} == {evidence[ev]} & {target} == {val}").shape[0]/self.data.shape[0])/self.prob_tables[target][val])
        alpha = 1/sum(new_probs.values())
        for val in new_probs:
            new_probs[val] *= alpha
        df = pd.Series(new_probs).sort_index()
        return df.copy()

app = Flask(__name__)

@app.route('/', methods=['GET'])
def main():
        return render_template('index.html')

@app.route('/classificator', methods=['GET'])
def classificator():
        return render_template('classificator.html')

@app.route('/result', methods=['GET'])
def result():
        in_d = {}
        for name in ['did_activity', 'did_interaction', 'is_AM', 'is_adult', 'on_ground']:
                in_d[name] = request.args.get(name)
        model = NaiveBayes(pd.read_csv('input.csv'))

        output = model.predict(in_d, 'color')
        return render_template('result.html',data=dict(output))
