import numpy as np
import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.datasets import fetch_california_housing


def main(verbosity=False):
    st.header("YouDo 1 - Mixture Regression Model with Loss calculation")
    st.markdown("""
    This is a modified version of the we do session #1 that includes loss calculation.

    The we do session includes an implementation of mixture linear regressor:
    
    1. A general linear regression model
    2. A **building age** level specific model.
    
    Ultimate loss function will be optimizing the parameters of both models at the same time
    """)

    st.header("Dataset")
    cal_housing = fetch_california_housing()
    X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    y = cal_housing.target

    st.dataframe(X)
    df = pd.DataFrame(
        dict(MedInc=X['MedInc'], Price=cal_housing.target, HouseAgeGroup=(X['HouseAge'].values / 10).astype(np.int)))

    st.dataframe(df)

    st.subheader("House Age independent General Model")
    fig = px.scatter(df, x="MedInc", y="Price", trendline="ols")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model per House Age Group")
    group = st.selectbox("House Age Group", [0, 1, 2, 3, 4, 5])
    fig = px.scatter(df[df["HouseAgeGroup"] == group], x="MedInc", y="Price", trendline="ols")
    st.plotly_chart(fig, use_container_width=True)

    if verbosity:
        st.subheader("Number of instance by House Age Group")
        st.dataframe(df.groupby('HouseAgeGroup').count())


    st.markdown("#### Loss function")

    st.latex(r"y = \beta_0 + \beta_1 x = {\beta} ^T x")
    st.write("""Find the **optimal** betas that will minimize my **error**
    """)

    st.latex(r"L(\beta_0, \beta_1) = \sum_{i=1}^{N}{(y_i - (\beta_0 + \beta_1 x_i))^2 }")

    loss, b0, b1 = [], [], []
    b0_space, b1_space = np.linspace(-10, 10, 50), np.linspace(-10, 10, 50)
    for _b0 in b0_space:
        loss_temp = []
        for _b1 in b1_space:
            b0.append(_b0)
            b1.append(_b1)

            loss_temp.append(np.power((df['Price'].values - _b1 * df['MedInc'].values - _b0), 2).mean())

        loss.append(loss_temp)

    loss = np.array(loss)
    loss_argmin = np.unravel_index(loss.argmin(), loss.shape)
    betas_for_min_loss = (b0_space[loss_argmin[0]], b1_space[loss_argmin[1]])
    st.write("Minimum loss at b[0]={} \
    and b[1]={}".format(betas_for_min_loss[0], betas_for_min_loss[1]))

    st.write("Loss is {} at that point.".format(loss[loss_argmin]))

    st.write("Plot of b1 when b0={}".format(betas_for_min_loss[0]))
    l = pd.DataFrame(dict(b1=b1_space, loss=loss[loss_argmin[0], :] ))
    fig = px.scatter(l, x="b1", y="loss")
    st.plotly_chart(fig, use_container_width=True)

    st.write("Plot of b0 when b1={}".format(betas_for_min_loss[1]))
    l = pd.DataFrame(dict(b0=b0_space, loss=loss[: , loss_argmin[1]] ))
    fig = px.scatter(l, x="b0", y="loss")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Formulating the mixture Group")

    st.markdown("#### General Model")
    st.latex(r"\hat{y}^{0}_i=\beta_0 + \beta_1 x_i")
    st.markdown("#### House Age Group specific Models")
    st.latex(r"\hat{y}^{1}_i=\gamma^{color}_0 + \gamma^{color}_1 x_i")

    st.markdown("#### Partial derivatives")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \beta_0}=-2p\sum_{i=1}^{N}{(y_i - \hat{y}_i) }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \beta_1}=-2p\sum_{i=1}^{N}{(y_i - \hat{y}_i)x_i }")

    st.write("Note that we calculate each group gradient separately")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \gamma^{color}_{0}}=-2 (1-p )\sum_{i \in HouseAgeGroup = color}{(y_i - \hat{y}_i) }")
    st.latex(
        r"\frac{\partial L(\beta_0,\beta_1,\gamma^{color}_{0},\gamma^{color}_{1})}{\partial \gamma^{color}_{1}}=-2(1-p )\sum_{i \in HouseAgeGroup = color}{(y_i - \hat{y}_i)x_i }")

    st.write(
        "**Mixture Ratio (p)** hyperparameter allows us to choose between pure HouseAgeGroup based model (`p=0`) and a common model over all instances (`p=1`)")

    p = st.slider("Mixture Ration (p)", 0.0, 1.0, value=0.8)
    beta, gamma = reg(df['MedInc'].values, df['Price'].values, df['HouseAgeGroup'].values,
                      p=p,
                      verbose=verbosity)

    




    st.subheader(f"General Model with p={p:.2f} contribution")
    st.latex(fr"Price = {beta[1]:.4f} \times MedInc + {beta[0]:.4f}")

    st.subheader(f"House Age Group Models with p={1 - p:.2f} contribution")
    st.latex(fr"Price = {gamma[0][1]:.4f} \times MedInc + {gamma[0][0]:.4f}")
    st.latex(fr"Price = {gamma[1][1]:.4f} \times MedInc + {gamma[1][0]:.4f}")
    st.latex(fr"Price = {gamma[2][1]:.4f} \times MedInc + {gamma[2][0]:.4f}")
    st.latex(fr"Price = {gamma[3][1]:.4f} \times MedInc + {gamma[3][0]:.4f}")
    st.latex(fr"Price = {gamma[4][1]:.4f} \times MedInc + {gamma[4][0]:.4f}")


def reg(x, y, group, p=0.3, verbose=False):
    beta = np.random.random(2)
    gamma = dict((k, np.random.random(2)) for k in range(6))

    if verbose:
        st.write(beta)
        st.write(gamma)
        st.write(x)

    alpha = 0.0035
    my_bar = st.progress(0.)
    n_max_iter = 100
    for it in range(n_max_iter):

        err = 0
        for _k, _x, _y in zip(group, x, y):
            y_pred = p * (beta[0] + beta[1] * _x) + (1 - p) * (gamma[_k][0] + gamma[_k][1] * _x)

            g_b0 = -2 * p * (_y - y_pred)
            g_b1 = -2 * p * ((_y - y_pred) * _x)

            # st.write(f"Gradient of beta0: {g_b0}")

            g_g0 = -2 * (1 - p) * (_y - y_pred)
            g_g1 = -2 * (1 - p) * ((_y - y_pred) * _x)

            beta[0] = beta[0] - alpha * g_b0
            beta[1] = beta[1] - alpha * g_b1

            gamma[_k][0] = gamma[_k][0] - alpha * g_g0
            gamma[_k][1] = gamma[_k][1] - alpha * g_g1

            err += (_y - y_pred) ** 2

        print(f"{it} - Beta: {beta}, Gamma: {gamma}, Error: {err}")
        my_bar.progress(it / n_max_iter)

    return beta, gamma


if __name__ == '__main__':
    main(st.sidebar.checkbox("verbosity"))