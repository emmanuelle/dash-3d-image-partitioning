import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
from skimage import data, img_as_ubyte, segmentation
from dash_canvas.utils import array_to_data_url
import plot_common
import image_utils
import numpy as np
from nilearn import image

def make_seg_image(img):
    """ Segment the image, then find the boundaries, then return an array that
    is clear where there are no boundaries. """
    segb=np.zeros_like(img).astype('uint8')
    ## segmentation not working in 3D? try segmenting each level...
    #for i in range(img.shape[0]):
    #    seg=segmentation.slic(img[i],start_label=1,multichannel=False, compactness=0.01,n_segments=300)
    #    segb[i,:,:]=segmentation.find_boundaries(seg).astype('uint8')
    seg=segmentation.slic(img,start_label=1,multichannel=False,compactness=0.1,n_segments=300)
    segb=segmentation.find_boundaries(seg).astype('uint8')
    segl=image_utils.label_to_colors(
    segb, colormap=['#000000','#E48F72'], alpha=[0,128], color_class_offset=0)
    #segl=image_utils.label_to_colors(seg)
    return segl

img = image.load_img("assets/BraTS19_2013_10_1_flair.nii")
img = img.get_data().transpose(2,0,1).astype('float')
img = img_as_ubyte((img - img.min())/(img.max() - img.min()))

# partition image
seg_img=img_as_ubyte(make_seg_image(img))
img_slices=[array_to_data_url(img[i, :, :]) for i in range(img.shape[0])]
seg_slices = [array_to_data_url(seg_img[i]) for i in range(seg_img.shape[0])] 

app = dash.Dash(__name__)

fig=figure=plot_common.dummy_fig()
fig=plot_common.add_layout_images_to_fig(fig,[img_slices[0],seg_slices[0]],img_args=dict(layer='above'),
width_scale=4,
height_scale=4,
update_figure_dims='height')

#print(fig.show('json'))

app.layout=html.Div([
    dcc.Store(id="image-slices",data=img_slices),
    dcc.Store(id="seg-slices",data=seg_slices),
    dcc.Slider(id="image-select-2",min=0,max=len(img_slices),step=1,updatemode="drag"),
    dcc.Checklist(
        id='show-seg-check',
        options=[
            {'label': 'Show segmentation', 'value': 'show'},
        ],
        value=['show']
    ),
    dcc.Graph(id="image-display-graph",figure=fig)
])

app.clientside_callback(
"""
function(
    image_select_value2,
    show_seg_check,
    image_slices_data,
    image_display_figure,
    seg_slices_data) {
    		console.log(image_display_figure);
    		var image_display_figure_ = {...image_display_figure};
    		image_display_figure_.layout.images[0].source=image_slices_data[image_select_value2];
            if (show_seg_check == 'show') {
                image_display_figure_.layout.images[1].source=seg_slices_data[image_select_value2];
            } else {
                image_display_figure_.layout.images[1].source="";
            }
            console.log(image_display_figure_);
    		return image_display_figure_;
		}
"""
,
    Output("image-display-graph","figure"),
    [Input("image-select-2","value"),
     Input("show-seg-check","value")],
    [State("image-slices","data"),
     State("image-display-graph","figure"),
     State("seg-slices","data")]
)

if __name__ == '__main__':
    app.run_server(debug=True)
