import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import json
from dash import Dash, dcc, html, Input, Output

# Load data
df = pd.read_csv("dataset_olympics.csv")

def process_null_values(df):
    # Fill null values in 'Medal' column with 'N'
    df['Medal'] = df['Medal'].fillna('N')

    # Define groupings for progressive filling
    groupings = [
        ['NOC', 'Sport', 'Year','Sex','Medal'],
        ['Sport', 'Year','Sex', 'Medal'],
        ['Sport', 'Year','Sex'],
        ['Sport','Sex']
    ]

    # Function to fill null values for a given column
    def fill_nulls(column):
        for group in groupings:
            avg = df.groupby(group)[column].transform('mean')
            df.loc[df[column].isna(), column] = avg

    # Process Age, Height, and Weight
    for column in ['Age', 'Height', 'Weight']:
        fill_nulls(column)

    return df

df = process_null_values(df)

# Read the JSON file
with open('bounding_boxes_data.json', 'r') as f:
    data = json.load(f)

# Extract the bounding boxes and max values
max_values = data['max_values']
bounding_boxes = {}
sports = ['Gymnastics', 'Swimming', 'Athletics', 'Shooting', 'Sailing', 'Fencing', 'Rowing', 'Boxing', 'Equestrianism',
          'Weightlifting', 'Cycling', 'Diving', 'Basketball', 'Football', 'Wrestling']

for sport in sports:
    if sport in data['bounding_boxes']:
        bounding_boxes[sport] = data['bounding_boxes'][sport]['best_box']
        overall_medal_prob = data['bounding_boxes'][sport]['overall_medal_prob']
        total_athletes = data['bounding_boxes'][sport]['total_athletes']

def create_dash_app():
    # Initialize the Dash app
    app = Dash(__name__)

    # Define the layout
    app.layout = html.Div([
        html.H1("Olympic Sports Dashboard", style={'textAlign': 'center', 'marginTop': '20px'}),
        html.Div([
            html.Div([
                html.Label("Select Sport:"),
                dcc.Dropdown(
                    id='sport-dropdown',
                    options=[{'label': sport, 'value': sport} for sport in sports],
                    value=sports[0],
                    style={'width': '100%'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                html.Label("Select Medals:"),
                dcc.Checklist(
                    id='medal-checklist',
                    options=[
                        {'label': 'Gold', 'value': 'Gold'},
                        {'label': 'Silver', 'value': 'Silver'},
                        {'label': 'Bronze', 'value': 'Bronze'},
                        {'label': 'N', 'value': 'N'}
                    ],
                    value=['Gold', 'Silver', 'Bronze', 'N'],
                    inline=True
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                html.Label("Show Bounding Box:"),
                dcc.Checklist(
                    id='show-bounding-box',
                    options=[{'label': 'Show', 'value': 'show'}],
                    value=[],
                    inline=True
                )
            ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'})
        ], style={'padding': '20px', 'marginBottom': '20px'}),
        html.Div([
            dcc.Graph(id='3d-scatter-plot', style={'height': '80vh'})
        ], style={'padding': '20px'})
    ], style={'padding': '20px', 'maxWidth': '1400px', 'margin': 'auto'})

    @app.callback(
        Output('3d-scatter-plot', 'figure'),
        [Input('sport-dropdown', 'value'),
         Input('medal-checklist', 'value'),
         Input('show-bounding-box', 'value')]
    )
    def update_graph(selected_sport, selected_medals, show_bounding_box):
        # Filter the DataFrame for the selected sport and sex (Male)
        sport_df = df[(df['Sport'] == selected_sport) & (df['Sex'] == 'M')]
        total_sample_size = len(sport_df)
        # Filter for selected medals (this won't affect the stats calculation)
        filtered_df = sport_df[sport_df['Medal'].isin(selected_medals)]

        # Create the 3D scatter plot
        fig = px.scatter_3d(filtered_df,
                            x='Age',
                            y='Height',
                            z='Weight',
                            color='Medal',
                            color_discrete_map={'Gold': 'gold', 'Silver': 'silver', 'Bronze': '#cd7f32', 'N': 'blue'},
                            hover_name='Name',
                            hover_data=['Year'],
                            title=f'Age, Height, and Weight Distribution for {selected_sport} (Male)')

        # Set fixed axis ranges based on max values for the selected sport
        fig.update_layout(
            scene=dict(
                xaxis_title='Age',
                yaxis_title='Height',
                zaxis_title='Weight',
                xaxis=dict(range=[0, max_values[selected_sport]['Age']]),
                yaxis=dict(range=[0, max_values[selected_sport]['Height']]),
                zaxis=dict(range=[0, max_values[selected_sport]['Weight']])
            ),
            legend_title='Medal',
            margin=dict(l=0, r=0, b=0, t=40),
            height=700,
            uirevision='constant'  # This will keep the camera angle constant
        )

        # Add bounding box if checkbox is selected
        if show_bounding_box and 'show' in show_bounding_box and bounding_boxes[selected_sport] is not None:
            box = bounding_boxes[selected_sport]
            x_min, y_min, z_min = box['min_bounds_original']
            x_max, y_max, z_max = box['max_bounds_original']

            # Create cube
            cube = go.Mesh3d(
                x=[x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_min],
                y=[y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max],
                z=[z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max],
                i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                color='rgba(255, 0, 0, 0.2)',  # Semi-transparent red
                opacity=0.6,
                name='Bounding Box'
            )
            fig.add_trace(cube)

            
            

            # Create stats text
            stats_text = (
             f"Bounding Box Statistics:<br>"
             f"Age: {x_min:.1f} - {x_max:.1f} years<br>"
             f"Height: {y_min:.1f} - {y_max:.1f} cm<br>"
             f"Weight: {z_min:.1f} - {z_max:.1f} kg<br>"
             f"Sample size in box: {box['size']}<br>"
             f"Total Population size: {total_sample_size}<br>"
             f"% in box: {(box['size'] / total_sample_size) * 100:.1f}%<br>"
             f"Medal Prob in Box: {box['medal_prob']:.2%}<br>"
             f"Overall Medal Prob: {overall_medal_prob:.2%}<br>"
             f"Improvement: {box['improvement']:.2f}x"
            )

            # Add annotation for bounding box statistics
            fig.add_annotation(
                x=0.05,  # Relative x-position on the plot
                y=0.95,  # Relative y-position on the plot
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                font=dict(size=14),
                align="left",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="red",
                borderwidth=2,
                borderpad=4
            )

        return fig

    return app

# Create the Dash app
app = create_dash_app()

# Get the Flask server
server = app.server

if __name__ == '__main__':
    # Get port from environment variable or use 8050 as default
    port = int(os.environ.get('PORT', 8050))
    app.run_server(host='0.0.0.0', port=port)
