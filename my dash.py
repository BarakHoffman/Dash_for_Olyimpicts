import pandas as pd
import numpy as np
import matplotlib as mtp
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
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

df= process_null_values(df)

from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go


def find_best_bounding_box(sport_df, n_divisions=5, random_state=42, n_initializations=50, n_refinements=25):
    # Extract the features: Age, Height, and Weight
    features = sport_df[['Age', 'Height', 'Weight']].values

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(features)

    # Create a boolean array for whether an athlete won a medal (Gold, Silver, or Bronze)
    all_medals = np.isin(sport_df['Medal'].values, ['Gold', 'Silver', 'Bronze'])

    # Total number of athletes in the sport
    total_athletes = len(sport_df)

    # Overall medal probability for all athletes in this sport
    overall_medal_prob = np.mean(all_medals)

    # Initialize variables to track the best bounding box and its medal probability
    best_box = None
    best_medal_prob = 0

    # Set the main random state for reproducibility
    main_random_state = np.random.RandomState(random_state)

    # Loop over initializations to find the best bounding box
    for i in range(n_initializations):
        # Use a different seed for each initialization
        seed = main_random_state.randint(1, 10000)
        random_state_for_init = np.random.RandomState(seed)

        # Randomly initialize min and max bounds for the bounding box using the new random state
        min_bounds = random_state_for_init.uniform(low=normalized_data.min(axis=0), high=normalized_data.max(axis=0))
        max_bounds = random_state_for_init.uniform(low=min_bounds, high=normalized_data.max(axis=0))

        # Perform a number of refinements to improve the bounding box
        for _ in range(n_refinements):
            # Loop over each dimension (Age, Height, Weight)
            for dim in range(3):
                # Try adjusting both the min and max bounds
                for bound in ['min', 'max']:
                    # Original value of the current bound (either min or max)
                    original_value = min_bounds[dim] if bound == 'min' else max_bounds[dim]

                    # Calculate the step size for changing the bounds
                    step = (max_bounds[dim] - min_bounds[dim]) / n_divisions

                    # Generate new values by shifting the bound in both directions
                    new_values = original_value + np.arange(-1, 2) * step

                    # Test each new bound value
                    for new_value in new_values:
                        # If adjusting the min bound
                        if bound == 'min':
                            temp_min_bounds = min_bounds.copy()
                            temp_min_bounds[dim] = new_value
                            temp_max_bounds = max_bounds
                        # If adjusting the max bound
                        else:
                            temp_max_bounds = max_bounds.copy()
                            temp_max_bounds[dim] = new_value
                            temp_min_bounds = min_bounds

                        # Create a mask to check which athletes fall inside the bounding box
                        in_box_mask = np.all(
                            (normalized_data >= temp_min_bounds) & (normalized_data <= temp_max_bounds), axis=1)
                        in_box_size = np.sum(in_box_mask)

                        # Ensure the box size is between 3% and 25% of total athletes
                        if 0.03 <= in_box_size / total_athletes <= 0.25:
                            # Calculate the probability of winning a medal within this bounding box
                            in_box_medals = all_medals[in_box_mask]
                            medal_prob = np.mean(in_box_medals)

                            # If this box has a better medal probability than the current best, update the best box
                            if medal_prob > best_medal_prob:
                                best_medal_prob = medal_prob
                                best_box = {
                                    'min_bounds': temp_min_bounds,
                                    'max_bounds': temp_max_bounds,
                                    'min_bounds_original': scaler.inverse_transform(temp_min_bounds.reshape(1, -1))[0],
                                    'max_bounds_original': scaler.inverse_transform(temp_max_bounds.reshape(1, -1))[0],
                                    'size': in_box_size,
                                    'medal_prob': medal_prob,
                                    'improvement': medal_prob / overall_medal_prob,
                                    'in_box_mask': in_box_mask
                                }

                    # After testing new values, update min or max bounds accordingly
                    if bound == 'min':
                        min_bounds = best_box['min_bounds'] if best_box else min_bounds
                    else:
                        max_bounds = best_box['max_bounds'] if best_box else max_bounds

    # Return the best bounding box, overall medal probability, and total athletes
    return best_box, overall_medal_prob, total_athletes


# Calculate max values and best bounding boxes for each sport
max_values = {}
bounding_boxes = {}
sports = ['Gymnastics', 'Swimming', 'Athletics', 'Shooting', 'Sailing', 'Fencing', 'Rowing', 'Boxing', 'Equestrianism', 'Weightlifting', 'Cycling', 'Diving', 'Basketball', 'Football', 'Wrestling']
for sport in sports:
    sport_df = df[(df['Sport'] == sport) & (df['Sex'] == 'M')]
    max_values[sport] = {
        'Age': sport_df['Age'].max(),
        'Height': sport_df['Height'].max(),
        'Weight': sport_df['Weight'].max()
    }
    best_box, overall_medal_prob, total_athletes = find_best_bounding_box(sport_df)
    bounding_boxes[sport] = best_box

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


# Define the callback to update the graph
@app.callback(
    Output('3d-scatter-plot', 'figure'),
    [Input('sport-dropdown', 'value'),
     Input('medal-checklist', 'value'),
     Input('show-bounding-box', 'value')]
)
def update_graph(selected_sport, selected_medals, show_bounding_box):
    # Filter the DataFrame for the selected sport and sex (Male)
    filtered_df = df[(df['Sport'] == selected_sport) & (df['Sex'] == 'M')]

    # Replace None with 'N' in the Medal column
    filtered_df['Medal'] = filtered_df['Medal'].fillna('N')

    # Filter for selected medals
    filtered_df = filtered_df[filtered_df['Medal'].isin(selected_medals)]

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
        height=700
    )

    # Add bounding box if checkbox is selected
    if show_bounding_box and bounding_boxes[selected_sport] is not None:
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

        # Calculate total sample size
        total_sample_size = len(filtered_df)

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
            font=dict(size=12),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="red",
            borderwidth=2,
            borderpad=4
        )

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8051)