import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

class ModelTesting:
    def __init__(
        self,
        model
    ):
        self.model=model
        

    def test(self,test_X,test_Y):
        ypreds=self.model.predict(test_X)

        preds=test_Y.to_frame("yreal").assign(ypreds=ypreds)
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

        return preds,score