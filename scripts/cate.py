import statsmodels.api as sm

def estimate_cate(R, ite):
    # add interecpt
    R = sm.add_constant(R)
    R.rename(columns={"const": "ATE"}, inplace=True)

    model = sm.OLS(endog = ite, 
                   exog = R)
    results = model.fit()
    return results