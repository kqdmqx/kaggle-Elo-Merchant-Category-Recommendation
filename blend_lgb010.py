import pandas as pd


def main():
    lgb010 = pd.read_csv("./data/submissions/lgb010.csv")
    public_tm = pd.read_csv("./data/submissions/results-tm/blend3.csv")
    public_wnsb = pd.read_csv("./data/submissions/results-wnsb/ens.csv")

    print(lgb010.shape)
    print(public_tm.shape)
    print(public_wnsb.shape)

    # lgb010

    df_base = pd.DataFrame({
        "lgb010": lgb010.set_index("card_id").target,
        "public_tm": public_tm.set_index("card_id").target,
        "public_wnsb": public_wnsb.set_index("card_id").target,
    })

    print(df_base.iloc[:, :].corr())

    df_base["target"] = df_base.lgb010 * .5 + df_base.public_tm * .5
    df_base[["target"]].to_csv("./data/submissions/blend010p.csv")


if __name__ == '__main__':
    main()
