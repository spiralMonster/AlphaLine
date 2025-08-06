import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from multiple_time_series_cv import MultipleTimeSeriesCV


class TrainModel:
    def __init__(
        self,
        model,
        training_period,
        test_period,
        cv_splits,
        lookahead,
        target
    ):
        self.model=model
        self.target=target
        
        self.cv=MultipleTimeSeriesCV(
            training_period=training_period,
            test_period=test_period,
            lookahead=lookahead,
            n_splits=cv_splits
        )

        self.predictions=[]
        self.scores=[]


    def train(self,X,Y):
        for train_idx,val_idx in self.cv.split(X=X):
            train_X=X.iloc[train_idx]
            val_X=X.iloc[val_idx]

            train_Y=Y.iloc[train_idx][self.target]
            val_Y=Y.iloc[val_idx][self.target]

            self.model.fit(X=train_X,y=train_Y)

            ypreds=self.model.predict(val_X)

            preds=val_Y.to_frame("yreal").assign(ypreds=ypreds)
            preds_by_day=preds.groupby("date")

            ic=(preds_by_day
                .apply(lambda x: spearmanr(x.yreal,x.ypreds)[0]*100)
                .to_frame("ic")
               )
            
            rmse=(preds_by_day
                  .apply(lambda x: np.sqrt(mean_squared_error(y_pred=x.ypreds,y_true=x.yreal)))
                  .to_frame("rmse")
                 )

            score=pd.concat([ic,rmse],axis=1)

            self.scores.append(score)
            self.predictions.append(preds)
            

        self.scores=pd.concat(self.scores)
        self.predictions=pd.concat(self.predictions)

        return self.model,self.predictions,self.scores
            
            