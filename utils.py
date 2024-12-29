import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import numpy as np

def create_animated_map(df, fps=2, video_filename='animated_map.mp4'):
    """
    Creates an animated map showing the movement of entities over time.

    Parameters:
    - df (pd.DataFrame): DataFrame containing location data with columns ['id', 'latitude', 'longitude', 'timestamp'].
    - fps (int): Frames per second for the animation.
    - video_filename (str): Name of the output video file (e.g., 'animated_map.mp4').

    Returns:
    - None: Saves the animation as an MP4 file.
    """

    # Validate DataFrame columns
    required_columns = {'id', 'latitude', 'longitude', 'timestamp'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    # Convert 'timestamp' to datetime if not already
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort the data by timestamp
    df = df.sort_values('timestamp')

    # Get unique timestamps sorted
    unique_timestamps = df['timestamp'].sort_values().unique()

    # Get unique entity IDs
    entity_ids = df['id'].unique()

    # Create a color map for entities
    colors = plt.cm.get_cmap('viridis', len(entity_ids))
    id_color_map = {entity_id: colors(i) for i, entity_id in enumerate(entity_ids)}

    # Prepare data grouped by timestamp
    data_by_time = [df[df['timestamp'] == timestamp] for timestamp in unique_timestamps]

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))

    # Determine plot boundaries with padding
    padding = 0.01
    min_lat = df['latitude'].min() - padding
    max_lat = df['latitude'].max() + padding
    min_lon = df['longitude'].min() - padding
    max_lon = df['longitude'].max() + padding

    ax.set_xlim(min_lon, max_lon)
    ax.set_ylim(min_lat, max_lat)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Animated Location Data Over Time')

    # Initialize scatter plot with empty data
    scatter = ax.scatter([], [], s=100, edgecolors='k')

    # Initialize time annotation
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Add a legend for entities
    for entity_id in entity_ids:
        ax.scatter([], [], color=id_color_map[entity_id], label=f'Entity {entity_id}', s=100, edgecolors='k')
    ax.legend(loc='upper right')

    def init():
        """Initialize the scatter plot and time annotation."""
        scatter.set_offsets(np.empty((0, 2)))  # Empty 2D array
        scatter.set_color([])
        time_text.set_text('')
        return scatter, time_text

    def animate(i):
        """Update the scatter plot and time annotation for frame i."""
        current_data = data_by_time[i]

        if current_data.empty:
            # Handle case with no data points
            coords = np.empty((0, 2))
            colors_list = []
            timestamp_str = unique_timestamps[i].strftime('%Y-%m-%d %H:%M:%S')
        else:
            # Extract coordinates
            coords = current_data[['longitude', 'latitude']].values
            # Ensure coords is 2D
            if coords.ndim == 1:
                coords = coords.reshape(1, 2)
            # Assign colors based on entity ID
            colors_list = [id_color_map[entity_id] for entity_id in current_data['id']]
            # Get current timestamp
            timestamp_str = current_data['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')

        # Update scatter plot
        scatter.set_offsets(coords)
        scatter.set_color(colors_list)

        # Update time annotation
        time_text.set_text(f'Time: {timestamp_str}')

        return scatter, time_text

    # Create the animation
    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(unique_timestamps), interval=1000/fps, blit=True
    )

    # Save the animation as MP4 video
    try:
        ani.save(video_filename, writer='ffmpeg', fps=fps)
        print(f"Animation saved as {video_filename}")
    except Exception as e:
        print(f"Failed to save MP4 video: {e}")

    plt.close(fig)  # Close the figure to free memory
    
    
