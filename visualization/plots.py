import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

COLORS = {
    'truth': '#2196F3',
    'indeterminacy': '#FF9800',
    'falsehood': '#F44336',
    'original': '#607D8B',
    'modified': '#4CAF50',
    'reject': '#E53935',
    'indeterminate_decision': '#FB8C00',
    'fail_to_reject': '#43A047',
    'background': '#FAFAFA',
    'card': '#FFFFFF',
    'accent': '#1565C0'
}

def plot_neutrosophic_boxplot(groups, group_names, title):
    fig = go.Figure()
    
    for i, (g, name) in enumerate(zip(groups, group_names)):
        t_mids = [(n.T[0]+n.T[1])/2 for n in g.data]
        f_lows = [n.F[0] for n in g.data]
        f_highs = [n.F[1] for n in g.data]
        i_lows = [n.I[0] for n in g.data]
        i_highs = [n.I[1] for n in g.data]
        
        # Primary box plot for Truth values
        fig.add_trace(go.Box(
            y=t_mids,
            name=name,
            marker_color=COLORS['truth'],
            boxpoints=False,
            legendgroup='Truth',
            showlegend=(i==0)
        ))
        
        # Overlay Indeterminacy bands globally or as scatter overlay
        # representing the indeterminacy spread around the truth mid
        scatter_y = t_mids
        fig.add_trace(go.Scatter(
            x=[name]*len(scatter_y),
            y=scatter_y,
            mode='markers',
            error_y=dict(
                type='data',
                symmetric=False,
                array=[h-0 for h in i_highs], # upper indeterminacy offset
                arrayminus=[0-l for l in i_lows], # lower indet offset
                color=COLORS['indeterminacy'],
                thickness=1.5,
                width=3
            ),
            marker=dict(color=COLORS['indeterminacy'], size=4, opacity=0.7),
            name='Indeterminacy',
            legendgroup='Indeterminacy',
            showlegend=(i==0)
        ))

    fig.update_layout(
        title=title, 
        plot_bgcolor=COLORS['background'], 
        paper_bgcolor=COLORS['background'],
        yaxis_title="Values"
    )
    return fig

def plot_power_curves(simulation_results: pd.DataFrame) -> go.Figure:
    if simulation_results.empty: return go.Figure()
    
    # We create a facet plot by distribution and indeterminacy_level manually
    # For a main dashboard chart, we'll plot power vs n, colored by variant
    fig = px.line(
        simulation_results, 
        x='n', y='power', 
        color='variant', 
        facet_col='distribution',
        line_dash='test',
        markers=True,
        title='Statistical Power Curves (n vs Power)',
        color_discrete_map={'original': COLORS['original'], 'modified': COLORS['modified']}
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_pvalue_interval(p_lower: float, p_upper: float, alpha: float = 0.05) -> go.Figure:
    fig = go.Figure()
    
    # Background zones
    fig.add_shape(type="rect", x0=0, y0=0, x1=alpha, y1=1, fillcolor=COLORS['reject'], opacity=0.1, line_width=0)
    fig.add_shape(type="rect", x0=alpha, y0=0, x1=1, y1=1, fillcolor=COLORS['fail_to_reject'], opacity=0.1, line_width=0)
    
    # Alpha line
    fig.add_vline(x=alpha, line_width=2, line_dash="dash", line_color=COLORS['reject'], annotation_text=f"α = {alpha}")
    
    # The actual p-value interval
    fig.add_trace(go.Scatter(
        x=[p_lower, p_upper], y=[0.5, 0.5],
        mode='lines+markers',
        line=dict(color=COLORS['indeterminacy'], width=8),
        marker=dict(size=12, color=COLORS['indeterminacy']),
        name='p-value interval'
    ))
    
    # Points for bounds if strictly different
    if p_lower != p_upper:
        fig.add_annotation(x=p_lower, y=0.55, text=f"{p_lower:.4f}", showarrow=False)
        fig.add_annotation(x=p_upper, y=0.55, text=f"{p_upper:.4f}", showarrow=False)
    
    fig.update_layout(
        title="Neutrosophic P-Value Interval",
        xaxis=dict(range=[0, 1.05], title="p-value"),
        yaxis=dict(showticklabels=False, range=[0, 1]),
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor=COLORS['background'],
        paper_bgcolor=COLORS['background']
    )
    return fig

def plot_contingency_heatmap(table: np.ndarray, row_labels: list, col_labels: list, title: str) -> go.Figure:
    fig = px.imshow(
        table, 
        labels=dict(y="Zone", x="Group", color="Count"),
        x=col_labels, y=row_labels,
        text_auto=True,
        color_continuous_scale='Blues',
        title=title
    )
    fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_dominance_triple(p_T: float, p_I: float, p_F: float) -> go.Figure:
    labels = ['P(X > Y) [Truth]', 'P(X ≈ Y) [Indeterminacy]', 'P(X < Y) [Falsehood]']
    values = [p_T, p_I, p_F]
    colors = [COLORS['truth'], COLORS['indeterminacy'], COLORS['falsehood']]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
    fig.update_layout(title_text="Neutrosophic Dominance Probability",
                      plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_radar_comparison(original_metrics: dict, modified_metrics: dict, test_name: str) -> go.Figure:
    categories = ['Power', 'Type I Control', 'Stability', 'Efficiency', 'Precision', 'Robustness']
    
    # dummy mappings since original metrics dict might just have power, type1 etc.
    def map_metrics(m):
        return [
            m.get('power', 0),
            1.0 - m.get('type1_error', 0), # Higher is better control
            m.get('decision_stability', 0),
            m.get('relative_efficiency', 1.0) / 2.0, # scale it
            1.0 - np.clip(m.get('interval_width', 0), 0, 1),
            m.get('power', 0) * m.get('decision_stability', 0)
        ]
        
    orig_vals = map_metrics(original_metrics) if original_metrics else [0.5]*6
    mod_vals = map_metrics(modified_metrics) if modified_metrics else [0.6]*6
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=orig_vals,
        theta=categories,
        fill='toself',
        name='Original',
        marker_color=COLORS['original']
    ))
    fig.add_trace(go.Scatterpolar(
        r=mod_vals,
        theta=categories,
        fill='toself',
        name='Modified',
        marker_color=COLORS['modified']
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title=f"Radar Comparison: {test_name}",
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background']
    )
    return fig

def plot_relative_efficiency(re_df: pd.DataFrame) -> go.Figure:
    if re_df.empty: return go.Figure()
    re_df['condition'] = "n=" + re_df['n'].astype(str) + " (" + re_df['distribution'] + ")"
    
    fig = px.bar(
        re_df[re_df['variant'] == 'modified'], 
        x='condition', y='relative_efficiency',
        color='relative_efficiency',
        color_continuous_scale=[(0, COLORS['truth']), (0.5, 'white'), (1, COLORS['indeterminacy'])],
        range_color=[0.8, 1.2],
        title='Relative Efficiency (Modified / Original)'
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS['falsehood'])
    fig.update_layout(yaxis_title="RE (Values > 1 indicate improvement)",
                      plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_type1_heatmap(type1_df: pd.DataFrame) -> go.Figure:
    if type1_df.empty: return go.Figure()
    
    pivot = type1_df.pivot_table(index='n', columns='delta', values='type1_error', aggfunc='mean')
    
    fig = px.imshow(
        pivot, 
        labels=dict(x="Indeterminacy Rate (δ)", y="Sample Size (n)", color="Type I Error"),
        x=pivot.columns, y=pivot.index,
        color_continuous_scale='RdYlGn_r',
        color_continuous_midpoint=0.05,
        text_auto=".3f",
        title="Type I Error Map (Green ≤ 0.05)"
    )
    fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_decision_stability(stability_df: pd.DataFrame) -> go.Figure:
    if stability_df.empty: return go.Figure()
    
    fig = px.area(
        stability_df,
        x='delta', y='decision_stability', color='variant',
        title="Decision Stability over Indeterminacy Rate",
        color_discrete_map={'original': COLORS['original'], 'modified': COLORS['modified']}
    )
    fig.update_layout(plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_neutrosophic_timeseries(df: pd.DataFrame, time_col: str, value_col: str) -> go.Figure:
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df[time_col], y=df[f"{value_col}_T_lower"],
        mode='lines', line=dict(color=COLORS['truth']),
        name='Truth Trend'
    ))
    
    # Upper bound indeterminacy
    y_upper = df[f"{value_col}_T_lower"] + df[f"{value_col}_I_upper"]
    y_lower = df[f"{value_col}_T_lower"] - df[f"{value_col}_I_upper"]
    
    fig.add_trace(go.Scatter(
        x=df[time_col].tolist() + df[time_col].tolist()[::-1],
        y=y_upper.tolist() + y_lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 152, 0, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='Indeterminacy Band'
    ))
    
    fig.update_layout(title="Neutrosophic Time Series",
                      plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background'])
    return fig

def plot_neutrosophic_ranking(original_data: list, neutro_array) -> go.Figure:
    fig = go.Figure()
    # Simple line plot showing classical vs neutrosophic rank
    import scipy.stats
    crisp = [n.defuzzify() for n in original_data]
    classical_ranks = scipy.stats.rankdata(crisp, method='average')
    
    for i, (cr, nr) in enumerate(zip(classical_ranks, neutro_array.data)):
        nr_mid = (nr.T[0]+nr.T[1])/2
        nr_width = nr.I[1] - nr.I[0]
        
        fig.add_trace(go.Scatter(
            x=[cr, nr_mid],
            y=[i, i],
            mode='lines+markers',
            error_x=dict(type='data', array=[nr_width/2], arrayminus=[nr_width/2], symmetric=False),
            name=f"Obs {i}", showlegend=False
        ))
        
    fig.update_layout(
        title="Classical vs Neutrosophic Ranking",
        xaxis_title="Rank", yaxis_title="Observation Index",
        plot_bgcolor=COLORS['background'], paper_bgcolor=COLORS['background']
    )
    return fig
