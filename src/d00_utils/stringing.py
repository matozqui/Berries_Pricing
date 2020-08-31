# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
def get_comma_values(df_series):

    ##  Function to get all values from a dataframe series comma separated as a string    ##

    import re

    return (re.sub('[^A-Za-z0-9]+', "','", str(df_series.unique()))[2:-2])


# %%



