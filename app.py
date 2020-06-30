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
import nibabel as nib
import plotly.express as px
import shape_utils
from app_utils import get_env
from sys import exit
import io
import base64
import skimage

DEBUG_MASK=False
DEFAULT_STROKE_COLOR = px.colors.qualitative.Light24[0]
DEFAULT_STROKE_WIDTH = 5
# the scales for the top and side images (they might be different)
# TODO: If the width and height scales are different, strange things happen? For
# example I have observed the masks getting scaled unevenly, maybe because the
# axes are actually scaled evenly (fixed to the x axis?) but then the mask gets
# scaled differently?
hwscales = [(2,2),(2,2)]
# the number of dimensions displayed
NUM_DIMS_DISPLAYED = 2  # top and side
# the color of the triangles displaying the slice number
INDICATOR_COLOR = "DarkOrange"

# A string, if length non-zero, saves superpixels to this file and then exits
SAVE_SUPERPIXEL = get_env("SAVE_SUPERPIXEL", default="")
# A string, if length non-zero, loads superpixels from this file
LOAD_SUPERPIXEL = get_env("LOAD_SUPERPIXEL", default="")


def make_seg_image(img):
    """ Segment the image, then find the boundaries, then return an array that
    is clear (alpha=0) where there are no boundaries. """
    segb = np.zeros_like(img).astype("uint8")
    seg = segmentation.slic(
        img, start_label=1, multichannel=False, compactness=0.1, n_segments=300
    )
    segb = segmentation.find_boundaries(seg).astype("uint8")
    segl = image_utils.label_to_colors(
        segb, colormap=["#000000", "#E48F72"], alpha=[0, 128], color_class_offset=0
    )
    return (segl, seg)


def make_default_figure(
    images=[],
    stroke_color=DEFAULT_STROKE_COLOR,
    stroke_width=DEFAULT_STROKE_WIDTH,
    shapes=[],
    img_args=dict(layer="above"),
    width_scale=1,
    height_scale=1,
):
    fig = plot_common.dummy_fig()
    plot_common.add_layout_images_to_fig(
        fig,
        images,
        img_args=img_args,
        width_scale=width_scale,
        height_scale=height_scale,
        update_figure_dims="height",
    )
    # add an empty image with the same size as the greatest of the already added
    # images so that we can add computed masks clientside later
    mwidth, mheight = [
        max([im[sz] for im in fig["layout"]["images"]]) for sz in ["sizex", "sizey"]
    ]
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
            layer="above",
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
img = img.get_data().transpose(2, 0, 1).astype("float")
print("img.shape", img.shape)
img = img_as_ubyte((img - img.min()) / (img.max() - img.min()))

def make_empty_found_segments():
    """ fstc_slices is initialized to a bunch of images containing nothing (clear pixels) """
    found_segs_tensor = np.zeros_like(img)
    # convert to a colored image (but it will just be colored "clear") 
    fst_colored = image_utils.label_to_colors(
        found_segs_tensor,
        colormap=["#000000", "#8A2BE2"],
        alpha=[0, 128],
        color_class_offset=0,
    )
    fstc_slices = [
        [
            array_to_data_url(np.moveaxis(fst_colored, 0, j)[i])
            for i in range(np.moveaxis(fst_colored, 0, j).shape[0])
        ]
        for j in range(NUM_DIMS_DISPLAYED)
    ]
    return fstc_slices

if len(LOAD_SUPERPIXEL) > 0:
    # load partitioned image (to save time)
    if LOAD_SUPERPIXEL.endswith(".gz"):
        import gzip

        with gzip.open(LOAD_SUPERPIXEL) as fd:
            dat = np.load(fd)
            segl = dat["segl"]
            seg = dat["seg"]
    else:
        dat = np.load(LOAD_SUPERPIXEL)
        segl = dat["segl"]
        seg = dat["seg"]
else:
    # partition image
    segl, seg = make_seg_image(img)

if len(SAVE_SUPERPIXEL) > 0:
    np.savez(SAVE_SUPERPIXEL, segl=segl, seg=seg)
    exit(0)

seg_img = img_as_ubyte(segl)
img_slices, seg_slices = [
    [
        # top
        [array_to_data_url(im[i, :, :]) for i in range(im.shape[0])],
        # side
        [array_to_data_url(im[:, i, :]) for i in range(im.shape[1])],
    ]
    for im in [img, seg_img]
]
# initially no slices have been found so we don't draw anything
found_seg_slices = make_empty_found_segments()

app = dash.Dash(__name__)
server=app.server

top_fig, side_fig = [
    make_default_figure(
        images=[img_slices[i][0], seg_slices[i][0]],
        width_scale=hwscales[i][1],
        height_scale=hwscales[i][0],
    )
    for i in range(NUM_DIMS_DISPLAYED)
]

# print(fig.show('json'))

app.layout = html.Div(
    [
        dcc.Store(id="image-slices", data=img_slices),
        dcc.Store(id="seg-slices", data=seg_slices),
        dcc.Store(
            id="drawn-shapes",
            data=[
                [[] for _ in range(seg_img.shape[i])] for i in range(NUM_DIMS_DISPLAYED)
            ],
        ),
        dcc.Store(id="slice-number-top", data=0),
        dcc.Store(id="slice-number-side", data=0),
        dcc.Store(id="found-segs", data=found_seg_slices),
        dcc.Store(
            id="undo-data",
            data=dict(
                undo_n_clicks=0,
                redo_n_clicks=0,
                undo_shapes=[],
                redo_shapes=[],
                # 2 arrays, one for each image-display-graph-{top,side}
                # each array contains the number of slices in that image view, and each
                # item of this array contains a list of shapes
                empty_shapes=[
                    [[] for _ in range(seg_img.shape[i])]
                    for i in range(NUM_DIMS_DISPLAYED)
                ],
            ),
        ),
        # the image data of the found segmentation is stored here before it is downloaded
        dcc.Store(id="found-image-tensor-data",data=""),
        # In this implementation we want to prevent needless passing of the
        # large image array from client to server, so when "downloaded-button"
        # is clicked, the contents of the "found-segs" store is converted to nii
        # imaging format, converted to base64, and stored in the
        # "found-image-tensor-data" store. When this store's contents are
        # updated, they are stored, decoded, in a Blob and a url is created from
        # the contents of this blob and set as the href of "download-link". Then
        # somehow we need to simulate a "click" on the "download-link". The
        # "found-image-tensor-data" store is necessary because we can only pass
        # base64-encoded data between client and server: we let the browser
        # handle how data from the browser can be written to the client's
        # filesystem.
        html.Button(
            "Download found partition", id="download-button"
        ),
        html.A(
            id="download-link",
            download="found_image.nii",
        ),
        # required so callback triggered by writing to "found-image-tensor-data"
        # has an output
        html.Div(
            id="dummy"
        ),
        dcc.Checklist(
            id="show-seg-check",
            options=[{"label": "Show segmentation", "value": "show"},],
            value=["show"],
        ),
        html.Button("Undo", id="undo-button", n_clicks=0),
        html.Button("Redo", id="redo-button", n_clicks=0),
        dcc.Graph(id="image-display-graph-top", figure=top_fig),
        html.Div(id="image-select-top-display"),
        dcc.Slider(
            id="image-select-top",
            min=0,
            max=len(img_slices[0]),
            step=1,
            updatemode="drag",
            value=0,
        ),
        dcc.Graph(id="image-display-graph-side", figure=side_fig),
        html.Div(id="image-select-side-display"),
        dcc.Slider(
            id="image-select-side",
            min=0,
            max=len(img_slices[1]),
            step=1,
            updatemode="drag",
            value=0,
        ),
    ]
)

app.clientside_callback(
    """
function(
    image_select_top_value,
    image_select_side_value,
    show_seg_check,
    found_segs_data,
    image_slices_data,
    image_display_top_figure,
    image_display_side_figure,
    seg_slices_data,
    drawn_shapes_data) {
    let image_display_figures_ = figure_display_update(
        [image_select_top_value,image_select_side_value],
        show_seg_check,
        found_segs_data,
        image_slices_data,
        [image_display_top_figure,image_display_side_figure],
        seg_slices_data,
        drawn_shapes_data),
        // slider order reversed because the image slice number is shown on the
        // other figure
        side_figure = image_display_figures_[1],
        top_figure = image_display_figures_[0],
        d=3,
        sizex, sizey;
    // append shapes that show what slice the other figure is in
    sizex = top_figure.layout.images[0].sizex,
    sizey = top_figure.layout.images[0].sizey;
    top_figure.layout.shapes=top_figure.layout.shapes.concat([
        tri_shape(d/2,sizey*image_select_side_value/found_segs_data[1].length,
                  d/2,d/2,'right'),
        tri_shape(sizex-d/2,sizey*image_select_side_value/found_segs_data[1].length,
                  d/2,d/2,'left'),
    ]);
    sizex = side_figure.layout.images[0].sizex,
    sizey = side_figure.layout.images[0].sizey;
    side_figure.layout.shapes=side_figure.layout.shapes.concat([
        tri_shape(d/2,sizey*image_select_top_value/found_segs_data[0].length,
                  d/2,d/2,'right'),
        tri_shape(sizex-d/2,sizey*image_select_top_value/found_segs_data[0].length,
                  d/2,d/2,'left'),
    ]);
    return image_display_figures_.concat(["Top image slice: " + image_select_top_value,
                                          "Side image slice: " + image_select_side_value,
                                          image_select_top_value,
                                          image_select_side_value
                                          ]);
}
""",
    [
        Output("image-display-graph-top", "figure"),
        Output("image-display-graph-side", "figure"),
        Output("image-select-top-display", "children"),
        Output("image-select-side-display", "children"),
        Output("slice-number-top", "data"),
        Output("slice-number-side", "data"),
    ],
    [
        Input("image-select-top", "value"),
        Input("image-select-side", "value"),
        Input("show-seg-check", "value"),
        Input("found-segs", "data"),
    ],
    [
        State("image-slices", "data"),
        State("image-display-graph-top", "figure"),
        State("image-display-graph-side", "figure"),
        State("seg-slices", "data"),
        State("drawn-shapes", "data"),
    ],
)

app.clientside_callback(
    """
function(top_relayout_data,
side_relayout_data,
undo_n_clicks,
redo_n_clicks,
top_slice_number,
side_slice_number,
drawn_shapes_data,
undo_data)
{
    drawn_shapes_data = json_copy(drawn_shapes_data);
    let ret = undo_track_slice_figure_shapes (
    [top_relayout_data,side_relayout_data],
    ["image-display-graph-top.relayoutData",
     "image-display-graph-side.relayoutData"],
    undo_n_clicks,
    redo_n_clicks,
    undo_data,
    drawn_shapes_data,
    [top_slice_number,side_slice_number],
    // a function that takes a list of shapes and returns those that we want to
    // track (for example if some shapes are to show some attribute but should not
    // be tracked by undo/redo)
    function (shapes) { return shapes.filter(function (s) {
            let ret = true;
            try { ret &= (s.fillcolor == "%s"); } catch(err) { ret &= false; }
            try { ret &= (s.line.color == "%s"); } catch(err) { ret &= false; }
            // return !ret because we don't want to keep the indicators
            return !ret;
        });
    });
    undo_data=ret[0];
    drawn_shapes_data=ret[1];
    return [drawn_shapes_data,undo_data];
}
"""
    % ((INDICATOR_COLOR,) * 2),
    [Output("drawn-shapes", "data"), Output("undo-data", "data")],
    [
        Input("image-display-graph-top", "relayoutData"),
        Input("image-display-graph-side", "relayoutData"),
        Input("undo-button", "n_clicks"),
        Input("redo-button", "n_clicks"),
    ],
    [
        State("slice-number-top", "data"),
        State("slice-number-side", "data"),
        State("drawn-shapes", "data"),
        State("undo-data", "data"),
    ],
)


@app.callback(
    Output("found-segs", "data"),
    [Input("drawn-shapes", "data")],
    [
        State("image-display-graph-top", "figure"),
        State("image-display-graph-side", "figure"),
    ],
)
def draw_shapes_react(
    drawn_shapes_data, image_display_top_figure, image_display_side_figure
):

    if any(
        [
            e is None
            for e in [
                drawn_shapes_data,
                image_display_top_figure,
                image_display_side_figure,
            ]
        ]
    ):
        return dash.no_update
    masks = np.zeros_like(img)
    for j, (graph_figure, (hscale, wscale)) in enumerate(
        zip([image_display_top_figure, image_display_side_figure], hwscales)
    ):
        fig = go.Figure(**graph_figure)
        # we use the width and the height of the first layout image (this will be
        # one of the images of the brain) to get the bounding box of the SVG that we
        # want to rasterize
        width, height = [fig.layout.images[0][sz] for sz in ["sizex", "sizey"]]
        for i in range(seg_img.shape[j]):
            shape_args = [
                dict(width=width, height=height, shape=s)
                for s in drawn_shapes_data[j][i]
            ]
            if len(shape_args) > 0:
                mask = shape_utils.shapes_to_mask(
                    shape_args,
                    # we only have one label class, so the mask is given value 1
                    1,
                )
                # TODO: Maybe there's a more elegant way to downsample the mask?
                np.moveaxis(masks, 0, j)[i, :, :] = mask[::hscale, ::wscale]
    found_segs_tensor = np.zeros_like(img)
    if DEBUG_MASK:
        found_segs_tensor[masks == 1] = 1
    else:
        # find labels beneath the mask
        labels = set(seg[1 == masks])
        # for each label found, select all of the segment with that label
        for l in labels:
            found_segs_tensor[seg == l] = 1
    # convert to a colored image
    fst_colored = image_utils.label_to_colors(
        found_segs_tensor,
        colormap=["#000000", "#8A2BE2"],
        alpha=[0, 128],
        color_class_offset=0,
    )
    fstc_slices = [
        [
            array_to_data_url(np.moveaxis(fst_colored, 0, j)[i])
            for i in range(np.moveaxis(fst_colored, 0, j).shape[0])
        ]
        for j in range(NUM_DIMS_DISPLAYED)
    ]
    return fstc_slices


def _decode_b64_slice(s):
    return base64.b64decode(s.encode())

# Converts found slices to nii file and encodes in b64 so it can be downloaded
def save_found_slices(fstc_slices):
    # we just save the first view (it makes no difference in the end)
    fstc_slices=fstc_slices[0]
    # convert encoded slices to array
    # TODO eventually make it format agnostic, right now we just assume png and
    # strip off length equal to uri_header from the uri string
    uri_header="data:image/png;base64,"
    # preallocating the final tensor by reading the first image makes converting
    # much faster (because all the images have the same dimensions)
    n_slices=len(fstc_slices)
    first_img=plot_common.str_to_img_ndarrary(
        _decode_b64_slice(fstc_slices[0][len(uri_header):]))
    fstc_ndarray=np.zeros((n_slices,)+first_img.shape,dtype=first_img.dtype)
    print('first_img.dtype',first_img.dtype)
    fstc_ndarray[0]=first_img
    for n,img_slice in enumerate(fstc_slices[1:]):
        img=plot_common.str_to_img_ndarrary(
            _decode_b64_slice(img_slice[len(uri_header):]))
        fstc_ndarray[n]=img
    print('fstc_ndarray.shape',fstc_ndarray.shape)
    # if the tensor is all zero (no partitions found) return None
    if np.all(fstc_ndarray == 0):
        return None
    # TODO add affine
    # technique for writing nii to bytes from here:
    # https://gist.github.com/arokem/423d915e157b659d37f4aded2747d2b3
    fstc_nii=nib.Nifti1Image(skimage.img_as_ubyte(fstc_ndarray),affine=None)
    fstcbytes = io.BytesIO()
    file_map = fstc_nii.make_file_map({'image':fstcbytes,'header':fstcbytes})
    fstc_nii.to_file_map(file_map)
    with open('/tmp/fstcbytes.nii','wb') as fd:
        fd.write(fstcbytes.getvalue())
    fstcb64 = base64.b64encode(fstcbytes.getvalue()).decode()
    # TODO: make clientside callback allowing user to download data of type
    # 'application/octet-stream'
    return fstcb64

@app.callback(
Output("found-image-tensor-data","data"),
[Input("download-button","n_clicks")],
[State("found-segs","data")])
def download_button_react(download_button_n_clicks,found_segs_data):
    ret = save_found_slices(found_segs_data)
    if ret is None:
        return ""
    return ret

app.clientside_callback(
"""
function (found_image_tensor_data) {
    if (found_image_tensor_data.length <= 0) {
        return "";
    }
    // for some reason you can't use the conversion to ascii from base64 directly
    // with blob, you have to use the ascii encoded as numbers
    const byte_chars = window.atob(found_image_tensor_data);
    const byte_numbers = Array.from(byte_chars,(b,i)=>byte_chars.charCodeAt(i));
    const byte_array = new Uint8Array(byte_numbers);
    let b = new Blob([byte_array],{type: 'application/octet-stream'});
    let url = URL.createObjectURL(b);
    return url;
}
"""
,
Output("download-link","href"),
[Input("found-image-tensor-data","data")])

app.clientside_callback(
"""
function (href) {
    if (href != "") {
        let download_a=document.getElementById("download-link");
        download_a.click();
    }
    return '';
}
""",
Output("dummy","children"),
[Input("download-link","href")])


if __name__ == "__main__":
    app.run_server(debug=True)
