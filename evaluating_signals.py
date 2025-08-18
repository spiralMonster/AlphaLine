import numpy as np
import pandas as pd
from alphalens.utils import get_clean_factor_and_forward_returns
from alphalens.tears import create_summary_tear_sheet


class Evaluating_Signals:
    def __init__(
        self,
        quantiles,
        periods,
        trade_prices
    ):
        self.quantiles=quantiles
        self.periods=periods
        self.trade_prices=trade_prices


    def evaluate_signal(self,model,feature_data):
        factor=model.predict(feature_data)
        factor=pd.Series(
            factor,
            index=pd.MultiIndex.from_arrays(
                [
                  feature_data.index.get_level_values("date"),
                  feature_data.index.get_level_values("ticker")  
                ],
                names=["date","ticker"]
            )
        )
        
        factor=(factor
                .sort_index()
                .tz_localize("UTC",level="date")
               )

        df=get_clean_factor_and_forward_returns(
            factor=factor,
            prices=self.trade_prices,
            quantiles=self.quantiles,
            periods=self.periods
        )

        return create_summary_tear_sheet(df)