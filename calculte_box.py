import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Load the dataset
df = pd.read_csv("dataset_olympics.csv")


def process_null_values(df):
    # Fill null values in 'Medal' column with 'N'
    df['Medal'] = df['Medal'].fillna('N')

    # Define groupings for progressive filling
    groupings = [
        ['NOC', 'Sport', 'Year', 'Sex', 'Medal'],
        ['Sport', 'Year', 'Sex', 'Medal'],
        ['Sport', 'Year', 'Sex'],
        ['Sport', 'Sex']
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

    # Initialize variables to track the best bounding box and its improvement
    best_box = None
    best_improvement = 1  # Initialize to 1 (no improvement)

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

                        # Transform the normalized bounds back to the original scale
                        temp_min_bounds_original = scaler.inverse_transform(temp_min_bounds.reshape(1, -1))[0]
                        temp_max_bounds_original = scaler.inverse_transform(temp_max_bounds.reshape(1, -1))[0]

                        # Create a mask to check which athletes fall inside the bounding box using sport_df
                        in_box_mask = (
                                (sport_df['Age'] >= temp_min_bounds_original[0]) &
                                (sport_df['Age'] <= temp_max_bounds_original[0]) &
                                (sport_df['Height'] >= temp_min_bounds_original[1]) &
                                (sport_df['Height'] <= temp_max_bounds_original[1]) &
                                (sport_df['Weight'] >= temp_min_bounds_original[2]) &
                                (sport_df['Weight'] <= temp_max_bounds_original[2])
                        )
                        in_box_size = np.sum(in_box_mask)

                        # Ensure the box size is between 3% and 25% of total athletes
                        if 0.03 <= in_box_size / total_athletes <= 0.25:
                            # Calculate the probability of winning a medal within this bounding box
                            in_box_medals = sport_df.loc[in_box_mask, 'Medal'].isin(['Gold', 'Silver', 'Bronze'])
                            medal_prob = np.mean(in_box_medals)

                            # Calculate improvement
                            improvement = medal_prob / overall_medal_prob

                            # If this box has a higher medal probability than overall and better improvement than the current best, update the best box
                            if medal_prob > overall_medal_prob and improvement > best_improvement:
                                best_improvement = improvement
                                best_box = {
                                    'min_bounds': temp_min_bounds,
                                    'max_bounds': temp_max_bounds,
                                    'min_bounds_original': temp_min_bounds_original,
                                    'max_bounds_original': temp_max_bounds_original,
                                    'size': in_box_size,
                                    'medal_prob': medal_prob,
                                    'improvement': improvement,
                                    'in_box_mask': in_box_mask
                                }

                    # After testing new values, update min or max bounds accordingly
                    if bound == 'min':
                        min_bounds = best_box['min_bounds'] if best_box else min_bounds
                    else:
                        max_bounds = best_box['max_bounds'] if best_box else max_bounds

    # Return the best bounding box, overall medal probability, and total athletes
    return best_box, overall_medal_prob, total_athletes


def calculate_and_save_bounding_boxes():
    sports = ['Gymnastics', 'Swimming', 'Athletics', 'Shooting', 'Sailing', 'Fencing', 'Rowing', 'Boxing',
              'Equestrianism', 'Weightlifting', 'Cycling', 'Diving', 'Basketball', 'Football', 'Wrestling']

    max_values = {}
    bounding_boxes = {}

    for sport in sports:
        sport_df = df[(df['Sport'] == sport) & (df['Sex'] == 'M')]
        max_values[sport] = {
            'Age': int(sport_df['Age'].max()),
            'Height': float(sport_df['Height'].max()),
            'Weight': float(sport_df['Weight'].max())
        }
        best_box, overall_medal_prob, total_athletes = find_best_bounding_box(sport_df)

        if best_box:
            best_box['min_bounds'] = best_box['min_bounds'].tolist()
            best_box['max_bounds'] = best_box['max_bounds'].tolist()
            best_box['min_bounds_original'] = best_box['min_bounds_original'].tolist()
            best_box['max_bounds_original'] = best_box['max_bounds_original'].tolist()
            best_box['in_box_mask'] = best_box['in_box_mask'].tolist()
            best_box['size'] = int(best_box['size'])
            best_box['medal_prob'] = float(best_box['medal_prob'])
            best_box['improvement'] = float(best_box['improvement'])

        bounding_boxes[sport] = {
            'box': best_box,
            'overall_medal_prob': float(overall_medal_prob),
            'total_athletes': int(total_athletes)
        }

    # Save to file
    with open('bounding_boxes_data.json', 'w') as f:
        json.dump({'max_values': max_values, 'bounding_boxes': bounding_boxes}, f, cls=NumpyEncoder)

    print("Bounding boxes calculated and saved to bounding_boxes_data.json")


if __name__ == "__main__":
    calculate_and_save_bounding_boxes()
