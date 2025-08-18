import numpy as np
import pandas as pd


class MultipleTimeSeriesCV:
    def __init__(
        self,
        training_period,
        test_period,
        n_splits,
        lookahead
    ):
        self.training_period=training_period
        self.test_period=test_period
        self.n_splits=n_splits
        self.lookahead=lookahead
        

    def split(self,X):
        unique_dates=X.index.get_level_values("date").unique()
        days=sorted(unique_dates,reverse=True)

        split_idx=[]
        do_break=False
        
        for ind in range(self.n_splits):
            test_end_idx=ind*self.test_period
            test_start_idx=test_end_idx+self.test_period

            train_end_idx=test_start_idx+self.lookahead-1
            train_start_idx=train_end_idx+self.training_period+self.lookahead-1

            if train_start_idx>=len(days):
                train_start_idx=len(days)-1
                do_break=True

            split_idx.append(
                [
                    train_start_idx,
                    train_end_idx,
                    test_start_idx,
                    test_end_idx
                ]
            )

            if do_break:
                break


        dates=X.reset_index()[["date"]]
        for train_start,train_end,test_start,test_end in split_idx:
            train_idx=dates[(dates.date>days[train_start]) & (dates.date<=days[train_end])].index
            test_idx=dates[(dates.date>days[test_start]) & (dates.date<=days[test_end])].index

            yield train_idx,test_idx

        



        