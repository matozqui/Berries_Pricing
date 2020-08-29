# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def normalize(df):

    ##  Function to normalize continous variables into 0 to 1 range ##

    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    norm_inst = MinMaxScaler(feature_range=(0.01, 0.99))
    norm = norm_inst.fit_transform(df)

    return norm_inst, norm


# %%
def denormalize(norm_inst,norm):

    ##  Function to denormalize continous variables into the real continous variable ##

    from sklearn.preprocessing import MinMaxScaler

    descaling_input = norm_inst.inverse_transform(norm)
    
    descaling_input =pd.DataFrame(descaling_input)
    descaling_input.index = norm.index
    descaling_input.columns = norm.columns.values

    return descaling_input


