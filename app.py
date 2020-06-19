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
import plotly.express as px
import shape_utils
from app_utils import get_env
from sys import exit

DEFAULT_STROKE_COLOR=px.colors.qualitative.Light24[0]
DEFAULT_STROKE_WIDTH=5
WIDTH_SCALE=4
HEIGHT_SCALE=4

# A string, if length non-zero, saves superpixels to this file and then exits
SAVE_SUPERPIXEL=get_env("SAVE_SUPERPIXEL",default="")
# A string, if length non-zero, loads superpixels from this file
LOAD_SUPERPIXEL=get_env("LOAD_SUPERPIXEL",default="")

def make_seg_image(img):
    """ Segment the image, then find the boundaries, then return an array that
    is clear (alpha=0) where there are no boundaries. """
    segb=np.zeros_like(img).astype('uint8')
    seg=segmentation.slic(img,start_label=1,multichannel=False,compactness=0.1,n_segments=300)
    segb=segmentation.find_boundaries(seg).astype('uint8')
    segl=image_utils.label_to_colors(
    segb, colormap=['#000000','#E48F72'], alpha=[0,128], color_class_offset=0)
    return (segl,seg)

def make_default_figure(
    images=[],
    stroke_color=DEFAULT_STROKE_COLOR,
    stroke_width=DEFAULT_STROKE_WIDTH,
    shapes=[],
    img_args=dict(layer='above')
):
    fig = plot_common.dummy_fig()
    plot_common.add_layout_images_to_fig(fig,
    images,img_args=img_args,width_scale=WIDTH_SCALE, height_scale=HEIGHT_SCALE,
    update_figure_dims='height')
    # add an empty image with the same size as the greatest of the already added
    # images so that we can add computed masks clientside later
    mwidth,mheight=[max([im[sz] for im in fig['layout']['images']]) for sz in ['sizex','sizey']]
    fig.add_layout_image(
            dict(
                source="",
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=mwidth,
                sizey=mheight,
                sizing="contain",
                layer="above"
                )
            )
    fig.update_layout(
        {
            "dragmode": "drawopenpath",
            "shapes": shapes,
            "newshape.line.color": stroke_color,
            "newshape.line.width": stroke_width,
            "margin": dict(l=0, r=0, b=0, t=0, pad=4),
        }
    )
    return fig

img = image.load_img("assets/BraTS19_2013_10_1_flair.nii")
img = img.get_data().transpose(2,0,1).astype('float')
img = img_as_ubyte((img - img.min())/(img.max() - img.min()))

if len(LOAD_SUPERPIXEL) > 0:
    # load partitioned image (to save time)
    dat=np.load(LOAD_SUPERPIXEL)
    segl=dat['segl']
    seg=dat['seg']
else:
    # partition image
    segl,seg=make_seg_image(img)

if len(SAVE_SUPERPIXEL) > 0:
    np.savez(SAVE_SUPERPIXEL,segl=segl,seg=seg)
    exit(0)

seg_img=img_as_ubyte(segl)
img_slices=[array_to_data_url(img[i, :, :]) for i in range(img.shape[0])]
seg_slices = [array_to_data_url(seg_img[i]) for i in range(seg_img.shape[0])] 
# initially no slices have been found so we don't draw anything
found_seg_slices=["" for i in range(seg_img.shape[0])]

app = dash.Dash(__name__)

fig=make_default_figure(images=[img_slices[0],seg_slices[0]])

#print(fig.show('json'))

app.layout=html.Div([
    dcc.Store(id="image-slices",data=img_slices),
    dcc.Store(id="seg-slices",data=seg_slices),
    dcc.Store(id="drawn-shapes",data=[[] for _ in seg_slices]),
    dcc.Store(id="slice-number",data=0),
    dcc.Store(id="found-segs",data=found_seg_slices),
    dcc.Slider(id="image-select",min=0,max=len(img_slices),step=1,updatemode="drag",value=0),
    html.Div(id='image-select-display'),
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
    image_select_value,
    show_seg_check,
    found_segs_data,
    image_slices_data,
    image_display_figure,
    seg_slices_data,
    drawn_shapes_data) {

    console.log(image_display_figure);
    var image_display_figure_ = json_copy(image_display_figure);
    if (!image_select_value) {
        // define if undefined
        image_select_value = 0;
    }
    image_display_figure_.layout.images[0].source=image_slices_data[image_select_value];
    if (show_seg_check == 'show') {
        image_display_figure_.layout.images[1].source=seg_slices_data[image_select_value];
        image_display_figure_.layout.images[2].source=found_segs_data[image_select_value];
    } else {
        image_display_figure_.layout.images[1].source="";
        image_display_figure_.layout.images[2].source="";
    }
    image_display_figure_.layout.shapes=drawn_shapes_data[image_select_value];
    console.log(image_display_figure_);
    return [image_display_figure_,image_select_value,image_select_value];
}
"""
,
    [Output("image-display-graph","figure"),
    Output("image-select-display","children"),
    Output("slice-number","data")],
    [Input("image-select","value"),
     Input("show-seg-check","value"),
     Input("found-segs","data")],
    [State("image-slices","data"),
     State("image-display-graph","figure"),
     State("seg-slices","data"),
     State("drawn-shapes","data")]
)

app.clientside_callback(
"""
function(graph_relayout_data, slice_number_data, drawn_shapes_data)
{
    console.log("relayoutData");
    console.log(graph_relayout_data);
    if (graph_relayout_data && ("shapes" in graph_relayout_data)) {
        drawn_shapes_data[slice_number_data]=graph_relayout_data.shapes;
    }
    return drawn_shapes_data;
}
""",
Output("drawn-shapes","data"),
[Input("image-display-graph","relayoutData")],
[State("slice-number","data"),
 State("drawn-shapes","data")])

@app.callback(
Output("found-segs","data"),
[Input("drawn-shapes","data")],
[State("image-display-graph","figure")])
def draw_shapes_react(drawn_shapes_data,image_display_graph_figure):
    
    if drawn_shapes_data is None or image_display_graph_figure is None:
        return dash.no_update
    fig=go.Figure(**image_display_graph_figure)
    # we use the width and the height of the first layout image (this will be
    # one of the images of the brain) to get the bounding box of the SVG that we
    # want to rasterize
    width,height=[fig.layout.images[0][sz] for sz in ['sizex','sizey']]
    masks=np.zeros_like(img)
    for i in range(seg_img.shape[0]):
        shape_args=[
        dict(width=width,
             height=height,
             shape=s) for s in drawn_shapes_data[i]]
        if len(shape_args) > 0:
            mask=shape_utils.shapes_to_mask(shape_args,
                 # we only have one label class, so the mask is given value 1
                 1)
            # TODO: Maybe there's a more elegant way to downsample the mask?
            masks[i,:,:]=mask[::HEIGHT_SCALE,::WIDTH_SCALE]
    import pdb
    #pdb.set_trace()
    found_segs_tensor=np.zeros_like(img)
    # find labels beneath the mask
    labels=set(seg[1==masks])
    # for each label found, select all of the segment with that label
    for l in labels:
        found_segs_tensor[seg==l]=1
    # convert to a colored image
    fst_colored=image_utils.label_to_colors(
        found_segs_tensor,
        colormap=['#000000','#8A2BE2'],
        alpha=[0,128],
        color_class_offset=0)
    fstc_slices=[array_to_data_url(fst_colored[i]) for i in range(fst_colored.shape[0])]
    return fstc_slices
    

if __name__ == '__main__':
    app.run_server(debug=True)
