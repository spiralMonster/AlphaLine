import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns


class Model_Analysis:
    def __init__(
        self,
        models,
        model_names,
        model_train_predictions,
        model_train_scores,
        test_X,
        test_Y
    ):
        self.models=models
        self.model_names=model_names
        self.model_train_predictions=model_train_predictions
        self.model_train_scores=model_train_scores
        self.model_testing_predictions=[]
        self.model_testing_scores=[]
        self.test_X=test_X
        self.test_Y=test_Y

        


    def get_training_analysis(self):
        print("[START] Training Analysis...")
        print("\n\n\n")

        print("[ANALYSIS] Jointplot between Predicted and Actual Values:")
        print("\n\n")
        self.get_jointplot(model_predictions=self.model_train_predictions,analysis_phase="Training")

        print("[ANALYSIS] IC Distribution:")
        print("\n\n")
        self.get_ic_distribution(model_scores=self.model_train_scores,analysis_phase="Training")

        print("[ANALYSIS] Rolling IC:")
        print("\n\n")
        self.get_rolling_ic(model_scores=self.model_train_scores,analysis_phase="Training")

        print("[ANALYSIS] Rolling RMSE:")
        print("\n\n")
        self.get_rolling_rmse(model_scores=self.model_train_scores,analysis_phase="Training")

        print("[END] Training Analysis...")
        


    
    
    def get_testing_analysis(self):
        self.get_model_predictions_and_scores(X=self.test_X,Y=self.test_Y)

        print("[START] Testing Analysis...")
        print("\n\n\n")

        print("[ANALYSIS] Jointplot between Predicted and Actual Values:")
        print("\n\n")
        self.get_jointplot(model_predictions=self.model_testing_predictions,analysis_phase="Testing")

        print("[ANALYSIS] IC Distribution:")
        print("\n\n")
        self.get_ic_distribution(model_scores=self.model_testing_scores,analysis_phase="Testing")

        print("[ANALYSIS] Rolling IC:")
        print("\n\n")
        self.get_rolling_ic(model_scores=self.model_testing_scores,analysis_phase="Testing")

        print("[ANALYSIS] Rolling RMSE:")
        print("\n\n")
        self.get_rolling_rmse(model_scores=self.model_testing_scores,analysis_phase="Testing")

        print("[END] Testing Analysis...")


    def get_combine_analysis(self):
        print("[START] Combine Analysis...")
        print("\n\n\n")

        print("[ANALYSIS] Rolling IC:")
        print("\n\n")
        self.get_combine_rolling_ic(training_scores=self.model_train_scores,testing_scores=self.model_testing_scores)

        print("[ANALYSIS] Rolling RMSE:")
        print("\n\n")
        self.get_combine_rolling_rmse(training_scores=self.model_train_scores,testing_scores=self.model_testing_scores)

        print("[END] Combine Analysis...")

    
    
    def get_model_predictions_and_scores(self,X,Y):
        
        for model in self.models:
            ypreds=model.predict(X)
            preds=Y.to_frame("yreal").assign(ypreds=ypreds)
            preds_by_day=preds.groupby("date")
            
            ic_by_day=(preds_by_day
                       .apply(lambda x: spearmanr(x.yreal,x.ypreds)[0]*100)
                       .to_frame("ic")
                      )

            rmse_by_day=(preds_by_day
                         .apply(lambda x: np.sqrt(mean_squared_error(y_true=x.yreal,y_pred=x.ypreds)))
                         .to_frame("rmse")
                        )

            score=pd.concat([ic_by_day,rmse_by_day],axis=1)

            self.model_testing_predictions.append(preds)
            self.model_testing_scores.append(score)
            
                    
        

        
    
    def get_jointplot(self,model_predictions,analysis_phase):
        for model_name,model_preds in zip(self.model_names,model_predictions):
            j=sns.jointplot(
                x="ypreds",
                y="yreal",
                joint_kws={
                    "line_kws":{"lw":1.0,"color":"k"},
                     "scatter_kws":{"s":1.0},
                    
                },
                kind="reg",
                data=model_preds
            )

            j.ax_joint.yaxis.set_major_formatter(
                FuncFormatter(lambda y,_: '{:.1%}'.format(y))
            )
            
            j.ax_joint.xaxis.set_major_formatter(
                FuncFormatter(lambda x,_: '{:.1%}'.format(x))
            )

            j.ax_joint.set_xlabel("Predicted Values")
            j.ax_joint.set_ylabel("Actual Values")

            title=analysis_phase+": "+model_name
            j.ax_joint.set_title(title)

            sns.despine()
            plt.tight_layout()
            plt.show()
            print("\n\n")


    def get_ic_distribution(self,model_scores,analysis_phase):
        for model_name,model_score in zip(self.model_names,model_scores):
            ax=sns.distplot(model_score.ic)
            ax.axvline(0,lw="1.0",ls="--",c="k")

            ic_mean=model_score.ic.mean(axis=0)
            ic_median=model_score.ic.median(axis=0)

            ax.text(
                x=0.05,
                y=0.9,
                s=f"Mean: {ic_mean}\nMedian: {ic_median}",
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes
            )

            ax.set_xlabel("Information Coefficient")

            title=analysis_phase+": "+model_name
            ax.set_title(title)

            sns.despine()
            plt.tight_layout()
            plt.show()
            print("\n\n")


    def get_rolling_ic(self,model_scores,analysis_phase,rolling_window=21):
        for model_name,model_scores in zip(self.model_names,model_scores):
            model_scores.ic.rolling(window=rolling_window).mean().plot(figsize=(16,5))

            ic_mean=model_scores.ic.mean(axis=0)
            plt.axhline(ic_mean,lw=1.0,ls="--",c="k")
            plt.axhline(0,lw=0.5,ls="-",c="k")

            plt.ylabel("IC")
            plt.legend()
            title=analysis_phase+": "+model_name
            plt.title(title)

            sns.despine()
            plt.tight_layout()
            plt.show()
            print("\n\n")


    def get_rolling_rmse(self,model_scores,analysis_phase,rolling_window=21):
        for model_name,model_score in zip(self.model_names,model_scores):
            model_score.rmse.rolling(window=rolling_window).mean().plot(figsize=(16,5),style=["g"])

            rmse_mean=model_score.rmse.mean(axis=0)
            plt.axhline(rmse_mean,lw=1.0,ls="--",c="k")

            plt.ylabel("RMSE")
            plt.legend()
            title=analysis_phase+": "+model_name
            plt.title(title)

            sns.despine()
            plt.tight_layout()
            plt.show()
            print("\n\n")


    def get_combine_rolling_ic(self,training_scores,testing_scores,rolling_window=21):
        for model_name,train_score,test_score in zip(self.model_names,training_scores,testing_scores):
            fig,ax=plt.subplots(figsize=(16,5))

            train_score.ic.rolling(window=rolling_window).mean().plot(ax=ax,style=["b"],label="Training")
            test_score.ic.rolling(window=rolling_window).mean().plot(ax=ax,style=["r"],label="Testing")

            ic_train_mean=train_score.ic.mean(axis=0)
            ic_test_mean=test_score.ic.mean(axis=0)

            ax.axhline(ic_train_mean,lw=1.0,ls="-",c="k",label="ic_mean_training")
            ax.axhline(ic_test_mean,lw=1.0,ls="--",c="k",label="ic_mean_testing")

            ax.set_ylabel("IC")
            ax.set_title(model_name)
            ax.legend()

            sns.despine()
            plt.tight_layout()
            plt.show()
            print("\n\n")

            

    def get_combine_rolling_rmse(self,training_scores,testing_scores,rolling_window=21):
        for model_name,train_score,test_score in zip(self.model_names,training_scores,testing_scores):
            fig,ax=plt.subplots(figsize=(16,5))

            train_score.rmse.rolling(window=rolling_window).mean().plot(ax=ax,style=["b"],label="Training")
            test_score.rmse.rolling(window=rolling_window).mean().plot(ax=ax,style=["r"],label="Testing")

            rmse_train_mean=train_score.rmse.mean(axis=0)
            rmse_test_mean=test_score.rmse.mean(axis=0)

            ax.axhline(rmse_train_mean,lw=1.0,ls="-",c="k",label="rmse_mean_training")
            ax.axhline(rmse_test_mean,lw=1.0,ls="--",c="k",label="rmse_mean_testing")

            ax.set_ylabel("RMSE")
            ax.set_title(model_name)
            ax.legend()

            sns.despine()
            plt.tight_layout()
            plt.show()
            print("\n\n")

        
            
            
            