from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.dependencies import ALL
import base64
import os
import re

app = Dash(__name__)
image_folder = '/home/juneyonglee/MyData/backup_20240914/AY_UST/ust21_modis_difference_map_loss_recon'
image_files = sorted(os.listdir(image_folder))

# Extract location and date information with error handling
image_info = []
for file in image_files:
    location = 'Nakdong' if 'nakdong' in file.lower() else 'Saemangeum' if 'saemangeum' in file.lower() else 'Unknown'
    date_match = re.search(r'\d{8}', file)
    date = date_match.group(0) if date_match else 'Unknown'

    image_info.append({
        'filename': file,
        'location': location,
        'date': date
    })

app.layout = html.Div([
    html.Div([
        html.Button('Nakdong', id='nakdong-btn', n_clicks=0, style={'margin-right': '10px'}),
        html.Button('Saemangeum', id='saemangeum-btn', n_clicks=0, style={'margin-right': '10px'}),
        html.Button('RMSE Nakdong', id='rmse-nakdong-btn', n_clicks=0, style={'margin-right': '10px'}),
        html.Button('RMSE Saemangeum', id='rmse-saemangeum-btn', n_clicks=0),
    ], style={'textAlign': 'center', 'margin': '20px'}),

    # Year selection dropdown
    html.Div([
        html.Label("Select Year:"),
        dcc.Dropdown(
            id='year-dropdown',
            options=[{'label': str(year), 'value': str(year)} for year in range(2012, 2022)],
            placeholder="Select a year",
            style={'width': '200px', 'margin': '0 auto'}
        )
    ], style={'textAlign': 'center', 'margin': '20px'}),

    # Interval component for automatic image transition
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0, disabled=True),

    # Image grid display
    html.Div(id='image-grid', style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}),
    html.Div(id='location-date-info', style={'textAlign': 'center', 'margin': '20px'}),
    html.Img(id='display-image', style={'width': '50%', 'display': 'block', 'margin': 'auto'})
])

@app.callback(
    Output('image-grid', 'children'),
    [Input('nakdong-btn', 'n_clicks'),
     Input('saemangeum-btn', 'n_clicks'),
     Input('year-dropdown', 'value')]
)
def update_image_grid(nakdong_clicks, saemangeum_clicks, selected_year):
    if nakdong_clicks > saemangeum_clicks:
        selected_location = 'Nakdong'
    elif saemangeum_clicks > nakdong_clicks:
        selected_location = 'Saemangeum'
    else:
        return []

    if selected_year is None:
        return []

    filtered_files = [
        info for info in image_info
        if info['location'] == selected_location and info['date'].startswith(selected_year)
    ]

    thumbnails = []
    for info in filtered_files:
        image_path = os.path.join(image_folder, info['filename'])
        encoded_thumbnail = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        thumbnails.append(html.Div(
            html.Img(
                src=f'data:image/png;base64,{encoded_thumbnail}',
                style={'width': '100px', 'height': '100px', 'padding': '5px', 'cursor': 'pointer'},
                id={'type': 'thumbnail', 'index': info['filename']}
            )
        ))

    return thumbnails

# Unified callback to control interval component and thumbnail click reset
@app.callback(
    [Output('interval-component', 'n_intervals'),
     Output('interval-component', 'disabled')],
    [Input('nakdong-btn', 'n_clicks'),
     Input('saemangeum-btn', 'n_clicks'),
     Input('rmse-nakdong-btn', 'n_clicks'),
     Input('rmse-saemangeum-btn', 'n_clicks'),
     Input({'type': 'thumbnail', 'index': ALL}, 'n_clicks')],
    [State('year-dropdown', 'value'), State('nakdong-btn', 'n_clicks'), State('saemangeum-btn', 'n_clicks')]
)
def control_interval(nakdong_clicks, saemangeum_clicks, rmse_nakdong_clicks, rmse_saemangeum_clicks, n_clicks_list, selected_year, nakdong_state, saemangeum_state):
    ctx = callback_context
    # Disable interval and reset if RMSE button is clicked
    if ctx.triggered and ctx.triggered[0]['prop_id'] in ['rmse-nakdong-btn.n_clicks', 'rmse-saemangeum-btn.n_clicks']:
        return 0, True

    # Enable interval from the beginning if location button is clicked
    if nakdong_clicks > 0 or saemangeum_clicks > 0:
        return 0, False  # Start from the beginning

    # Resume from the clicked thumbnail if a thumbnail is clicked
    if any(n_clicks_list):
        selected_location = 'Nakdong' if nakdong_state > saemangeum_state else 'Saemangeum'
        filtered_files = [
            info for info in image_info
            if info['location'] == selected_location and info['date'].startswith(selected_year)
        ]
        thumbnail_idx = max((i for i, n in enumerate(n_clicks_list) if n), default=0)
        if thumbnail_idx < len(filtered_files):
            return thumbnail_idx, False

    return 0, True  # Default to stopping the interval if no triggers are matched

# Callback to display images sequentially and stop at the last image
@app.callback(
    [Output('display-image', 'src'),
     Output('location-date-info', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('rmse-nakdong-btn', 'n_clicks'),
     Input('rmse-saemangeum-btn', 'n_clicks')],
    [State('nakdong-btn', 'n_clicks'),
     State('saemangeum-btn', 'n_clicks'),
     State('year-dropdown', 'value')]
)
def display_full_image(n_intervals, rmse_nakdong_clicks, rmse_saemangeum_clicks, nakdong_clicks, saemangeum_clicks, selected_year):
    ctx = callback_context

    # Display RMSE scatter images if the respective buttons are clicked
    if ctx.triggered and ctx.triggered[0]['prop_id'] == 'rmse-nakdong-btn.n_clicks':
        rmse_nakdong_path = os.path.join(image_folder, 'rmse_scatter_nakdong.png')
        encoded_image = base64.b64encode(open(rmse_nakdong_path, 'rb').read()).decode('ascii')
        return f'data:image/png;base64,{encoded_image}', "Displaying RMSE Scatter for Nakdong"
    elif ctx.triggered and ctx.triggered[0]['prop_id'] == 'rmse-saemangeum-btn.n_clicks':
        rmse_saemangeum_path = os.path.join(image_folder, 'rmse_scatter_saemangeum.png')
        encoded_image = base64.b64encode(open(rmse_saemangeum_path, 'rb').read()).decode('ascii')
        return f'data:image/png;base64,{encoded_image}', "Displaying RMSE Scatter for Saemangeum"

    # Determine the selected location based on button clicks
    selected_location = None
    if nakdong_clicks > saemangeum_clicks:
        selected_location = 'Nakdong'
    elif saemangeum_clicks > nakdong_clicks:
        selected_location = 'Saemangeum'

    if not selected_location or not selected_year:
        return None, ""

    # Filter images for the selected location and year
    filtered_files = [
        info for info in image_info
        if info['location'] == selected_location and info['date'].startswith(selected_year)
    ]

    if not filtered_files or n_intervals >= len(filtered_files):
        return None, "End of images"  # Stop displaying if last image is reached

    # Display the current image based on the interval count
    filename = filtered_files[n_intervals]['filename']
    image_path = os.path.join(image_folder, filename)
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
    location_date_text = f"Displaying image: {filename} for {selected_location} on {filtered_files[n_intervals]['date']}"

    return f'data:image/png;base64,{encoded_image}', location_date_text

if __name__ == '__main__':
    app.run_server(debug=True, port=8070)
