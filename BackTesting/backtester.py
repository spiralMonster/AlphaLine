import os
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

from zipline import run_algorithm
from zipline.api import (attach_pipeline,
                         pipeline_output,
                         slippage,
                         commission,
                         set_slippage,
                         set_commission,
                         order_target,
                         order_target_percent,
                         time_rules,
                         date_rules,
                         schedule_function,
                         record)


from zipline.data import bundles
from zipline.pipeline import CustomFactor,Pipeline
from zipline.pipeline.data import DataSet,Column
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.domain import US_EQUITIES
from zipline.utils.run_algo import load_extensions
from zipline.utils.calendar_utils import get_calendar

load_extensions(
    default=True,
    extensions=[],
    strict=True,
    environ=None
)

N_quantiles=5


class Model_Data(DataSet):
    prediction=Column(dtype=float)
    domain=US_EQUITIES


class Model_Factor(CustomFactor):
    inputs=[Model_Data.prediction]
    window_length=1

    def compute(self,today,assets,out,preds):
        out[:]=preds


class Quantile_Factor(CustomFactor):
    inputs=[Model_Data.prediction]
    window_length=1

    def compute(self,today,assets,out,preds):
        data=pd.Series(preds[0],index=assets)

        quantile_data=pd.qcut(
            data.rank(method="first"),
            q=N_quantiles,
            labels=False
        )+1

        out[:]=quantile_data.values


class BackTester:
    def __init__(
        self,
        n_longs,
        n_shorts,
        min_positions,
        capital_base,
        commission_cost,
        commission_min_trade_cost,
        slippage_spread,
        predictions_file,
        predictions_column_name,
        trading_strategy: dict
    ):

        self.n_longs=n_longs
        self.n_shorts=n_shorts
        self.min_positions=min_positions
        self.capital_base=capital_base
        self.commission_cost=commission_cost
        self.commission_min_trade_cost=commission_min_trade_cost
        self.slippage_spread=slippage_spread
        self.predictions_file=predictions_file
        self.predictions_column_name=predictions_column_name
        self.trading_strategy=trading_strategy

        self.bundle=bundles.load("quandl")

        

    def load_model_predictions(self):
        prediction=pd.read_csv(
            self.predictions_file,
            parse_dates=["date"],
            index_col=["date","ticker"]
        )[self.predictions_column_name]

        tickers=prediction.index.get_level_values("ticker").unique().tolist()
        assets=self.bundle.asset_finder.lookup_symbols(tickers,as_of_date=None)

        ticker_sids=pd.Index([asset.sid for asset in assets],dtype="int64")
        ticker_map=dict(zip(tickers,ticker_sids))

        self.prediction=(prediction
                    .swaplevel()
                    .unstack("ticker")
                    .rename(columns=ticker_map)
                    .tz_localize(None)
                   )

        self.assets=assets
        print(f"INFO: Model predictions loaded...")


    def fill_missing_dates_values(self):
        missing_dates=[]
        calendar=get_calendar("XNYS")
        prediction=self.prediction

        pred_dates=prediction.index.unique().tolist()
        start_date=pred_dates[0]
        end_date=pred_dates[-1]

        for date in calendar.sessions:
            if date>=start_date and date<=end_date:
                if date not in pred_dates:
                    missing_dates.append(date)

            if date>end_date:
                break


        for date in missing_dates:
            prediction.loc[date]=None

        prediction=prediction.sort_index().ffill()
        self.prediction=prediction

        print(f"INFO: Missing Date Values added in Model Predictions...")


    def compute_signals(self):
        if self.trading_strategy["strategy"]=="returns_based":
            model_signals=Model_Factor()

            pipeline=Pipeline(
                columns={
                    "longs":model_signals.top(self.n_longs,mask=model_signals>0),
                    "shorts":model_signals.bottom(self.n_shorts,mask=model_signals<0)
                },
                screen=StaticAssets(self.assets)
            )

        else:
            model_signals=Model_Factor()
            quantile_signals=Quantile_Factor()

            quantile_long=self.trading_strategy["values"]["quantile_long"]
            quantile_short=self.trading_strategy["values"]["quantile_short"]

            pipeline=Pipeline(
                columns={
                    "longs":quantile_signals.top(self.n_longs,mask=quantile_signals.eq(quantile_long)),
                    "shorts":quantile_signals.top(self.n_shorts,mask=quantile_signals.eq(quantile_short))
                },
                screen=StaticAssets(self.assets)
            )

        return pipeline

    def initialize(self,context):
        context.n_longs=self.n_longs
        context.n_shorts=self.n_shorts
        context.min_positions=self.min_positions
        context.universe=self.assets

        set_commission(commission.PerShare(cost=self.commission_cost,
                                           min_trade_cost=self.commission_min_trade_cost))

        set_slippage(slippage.FixedSlippage(spread=self.slippage_spread))
        

        schedule_function(
            self.rebalance,
            date_rules.every_day(),
            time_rules.market_open(hours=1,minutes=30)   
        )

        schedule_function(
            self.record_vars,
            date_rules.every_day(),
            time_rules.market_close()
        )

        pipeline=self.compute_signals()
        attach_pipeline(pipeline,"signals")
        

    def before_trading_start(self,context,data):
        output=pipeline_output("signals")

        final_output=pd.concat([output["longs"].astype(int),
                                output["shorts"].astype(int).mul(-1)],axis=0)

        trades=(final_output
                .reset_index()
                .drop_duplicates()
                .set_index("index")
                .squeeze()
               )

        context.trades=trades


    def rebalance(self,context,data):
        trades=defaultdict(list)

        for stock,trade in context.trades.items():
            if not trade:
                order_target(stock,0)

            else:
                trades[trade].append(stock)

        context.longs=len(trades[1])
        context.shorts=len(trades[-1])

        if context.longs>self.min_positions and context.shorts>self.min_positions:
            for stock in trades[1]:
                order_target_percent(stock,1/context.longs)

            for stock in trades[-1]:
                order_target_percent(stock,-1/context.shorts)


    def record_vars(self,context,data):
        record(
            leverage=context.account.leverage,
            longs=context.longs,
            shorts=context.shorts
        )


    def run_backtesting(self):
        self.load_model_predictions()
        self.fill_missing_dates_values()

        self.model_data_loader={
            Model_Data.prediction:DataFrameLoader(Model_Data.prediction,self.prediction)
        }

        print(f"INFO: Model Data Loader created...")
        
        if self.trading_strategy["strategy"]=="returns_based":
            strategy="trading strategy-I"

        else:
             strategy="trading strategy-II"
            
        print(f"INFO: Backtesting started for {strategy}...")

        start_date=self.prediction.index.get_level_values("date").min()
        end_date=self.prediction.index.get_level_values("date").max()

        results=run_algorithm(
            start=start_date,
            end=end_date,
            initialize=self.initialize,
            before_trading_start=self.before_trading_start,
            capital_base=self.capital_base,
            data_frequency="daily",
            bundle="quandl",
            custom_loader=self.model_data_loader 
        )

        print(f"INFO: Backtesting completed...")

        return results

        
        