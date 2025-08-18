import numpy as np
import pandas as pd


class Compute_Factors:
    def __init__(
        self,
        models,
        model_names,
        feature_data
    ):
        self.models=models
        self.model_names=model_names
        self.feature_data=feature_data


    def compute_factors(self):
        print(f"[INFO] Factor Computation started...")
        print("\n\n")
        
        factors=[]
        for model,name in zip(self.models,self.model_names):
            model_factor=model.predict(self.feature_data)

            model_factor=pd.DataFrame(
                {
                    name+"_factor":model_factor
                },
                index=pd.MultiIndex.from_arrays(
                    [
                        self.feature_data.index.get_level_values("date"),
                        self.feature_data.index.get_level_values("ticker")
                    ],
                    names=["date","ticker"]
                )
            )

            model_factor=(model_factor
                          .sort_index()
                         )

            factors.append(model_factor)
            print(f"[INFO] {name} Factors computed...")

        factors=pd.concat(factors,axis=1)
        factors["Combined_factors"]=factors.mean(axis=1)
        
        print(f"[INFO] Combined Factors computed...")
        print("\n\n")
        print(f"[INFO] Factor Computation completed...")
        
        return factors