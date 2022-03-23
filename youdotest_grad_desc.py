import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.datasets import fetch_california_housing

def gradient_descent(X, y, initial_b=None, alpha=0.0001, no_max_iterations=1000, early_stopping_threshold=0):

    b_history, grad_b0_history, grad_b1_history = [], [], []

    if initial_b is None:
        b = np.random.random(2)
    else:
        b = initial_b

    stop_early = (early_stopping_threshold != 0)

    for i in range(no_max_iterations):
        y_pred = b[0] + b[1] * X

        grad_b0 = -2 * (y - y_pred).mean()
        grad_b1 = -2 * (X * (y - y_pred)).mean()

        b_history.append(np.copy(b))
        grad_b0_history.append(np.copy(grad_b0))
        grad_b1_history.append(np.copy(grad_b1))

        b_prev = np.copy(b)

        b[0] = b[0] - alpha * grad_b0
        b[1] = b[1] - alpha * grad_b1

        print("i: {}\tbeta: {}\tgrad_b0: {}\t grad_b1: {}".format(i, b, grad_b0, grad_b1))

        if stop_early and np.linalg.norm(b - b_prev) < early_stopping_threshold:
            print("Early stopping at iteration {}!".format(i))
            break
    
    b_history, grad_b0_history, grad_b1_history = np.array(b_history), np.array(grad_b0_history), np.array(grad_b1_history)
    return b, b_history, grad_b0_history, grad_b1_history

def main(verbose: bool = False):
    cal_housing = fetch_california_housing()
    X_original = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
    X = X_original["MedInc"].values
    y = cal_housing.target

    if verbose:
        st.markdown("#### Original X dataframe")
        st.dataframe(X_original)

        st.markdown("#### X used (MedInc)")
        st.dataframe(X)

        st.markdown("#### y column (Price)")
        st.dataframe(y)

    beta, beta_history, grad_b0_history, grad_b1_history = gradient_descent(X, y, alpha=0.0001, no_max_iterations=4000, early_stopping_threshold=0.000001)

    fig_data = pd.DataFrame(dict(b0=beta_history[:, 0], b1=beta_history[:, 1], g_b0=grad_b0_history, g_b1=grad_b1_history))
    no_it = len(beta_history)
    
    fig_b0 = px.scatter(fig_data, x=range(1, no_it+1), y="b0", title="Beta 0")
    st.plotly_chart(fig_b0, use_container_width=True)

    fig_b1 = px.scatter(fig_data, x=range(1, no_it+1), y="b1", title="Beta 1")
    st.plotly_chart(fig_b1, use_container_width=True)

    fig_g_b0 = px.scatter(fig_data, x=range(1, no_it+1), y="g_b0", title="Gradient of beta 0")
    st.plotly_chart(fig_g_b0, use_container_width=True)

    fig_g_b1 = px.scatter(fig_data, x=range(1, no_it+1), y="g_b1", title="Gradient of beta 1")
    st.plotly_chart(fig_g_b1, use_container_width=True)



if __name__ == '__main__':
    _verbose = st.sidebar.checkbox("Verbose")
    main(verbose=_verbose)