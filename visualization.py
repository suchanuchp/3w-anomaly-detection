from plotly.subplots import make_subplots
from tqdm import tqdm
import plotly.graph_objects as go


def plot_by_instances(df):
    start_idx = df[df.start == 1].index
    cols = [col for col in df.columns if col not in ['timestamp', 'start']]
    for i, start in tqdm(enumerate(start_idx)):
        fig = make_subplots(rows=len(cols), cols=1, shared_xaxes=True,
                            subplot_titles=cols)

        stop = start_idx[i + 1] if i != len(start_idx) - 1 else len(df)
        poi = df.iloc[start:stop]
        for icol, col in enumerate(cols):
            fig.append_trace(go.Scatter(x=poi.index, y=poi[col], mode='lines',
                                        name=col),
                             row=icol + 1, col=1)

        abnormal_ts = poi[poi['class'] != 0].index
        if len(abnormal_ts) > 1:
            fig.add_vrect(x0=abnormal_ts[0], x1=abnormal_ts[-1],
                          fillcolor="red", opacity=0.6, line_width=0)

        fig.show()
