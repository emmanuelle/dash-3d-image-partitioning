import dash
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_html_components as html
import dash_core_components as dcc
import debug_utils
import dash_utils

app=dash_utils.new_dash_app(__file__,verbose=True)
fig_top=debug_utils.test_image_figure()
fig_bottom=debug_utils.test_image_figure(color="#FF4500")

app.layout=html.Div(children=[
    dcc.Graph(id="graph-A",figure=fig_top,
                                    config={
                                        "modeBarButtonsToAdd": [
                                            "drawopenpath",
                                        ],

                                    },
    ),
    dcc.Graph(id="graph-B",figure=fig_bottom,
                                    config={
                                        "modeBarButtonsToAdd": [
                                            "drawopenpath",
                                        ],
                                    },
    ),
    html.Button("Undo",id="undo-button",n_clicks=0),
    html.Button("Redo",id="redo-button",n_clicks=0),
    dcc.Store("undo-data",data=dict(
        undo_n_clicks=0,
        redo_n_clicks=0,
        undo_shapes=[],
        redo_shapes=[],
        # 2 shape lists, one for A fig, other for B fig
        empty_shapes=[[],[]]
    )),
    html.Div(id="dummy")
])

# TODO: This doesn't work with erasing or editing shapes, just adding new shapes
app.clientside_callback(
"""
function (
graphA,
graphB,
undo_n_clicks,
redo_n_clicks,
undo_data,
graphA_fig,
graphB_fig) {
    let triggered = dash_clientside.callback_context.triggered.map(t => t['prop_id']);
    console.log("triggered"); console.log(triggered);
    console.log("graphA"); console.log(graphA);
    console.log("graphB"); console.log(graphB);
    console.log("undo_n_clicks"); console.log(undo_n_clicks);
    console.log("redo_n_clicks"); console.log(redo_n_clicks);
    console.log("undo_data"); console.log(undo_data);
    console.log("graphA_fig"); console.log(graphA_fig);
    console.log("graphB_fig"); console.log(graphB_fig);
    // Things that could happen:
    // Shape drawn / deleted in graph A or B
    // undo or redo pressed
    let new_graphA_fig = json_copy(graphA_fig),
        new_graphB_fig = json_copy(graphB_fig),
        shapesA = ((new_graphA_fig.layout.shapes == undefined) ? [] : new_graphA_fig.layout.shapes),
        shapesB = ((new_graphB_fig.layout.shapes == undefined) ? [] : new_graphB_fig.layout.shapes);
    console.log("HELLO?");
    if ([graphA,graphB,undo_n_clicks,redo_n_clicks].map(
            x=>x!=undefined).reduce((a,v)=>a&&v,true)) {
        console.log("are we doing anything?");
        if (UndoState_undo_clicked(undo_data,undo_n_clicks)) {
            console.log("undo clicked");
            let updated_shapes = UndoState_apply_undo(undo_data);
            shapesA=updated_shapes[0];
            shapesB=updated_shapes[1];
        } else if (UndoState_redo_clicked(undo_data,redo_n_clicks)) {
            console.log("redo clicked");
            let updated_shapes = UndoState_apply_redo(undo_data);
            shapesA=updated_shapes[0];
            shapesB=updated_shapes[1];
        } else {
            console.log("shape drawn, no?");
            // When returning the new graph figure with the shapes, this doesn't
            // seem to change the contents of the relayoutData of the graph that
            // was not drawn on.
            // Also if the graph is zoomed, there won't be any shapes in graphA
            // or graphB, but we want the shapes to persist. So what we do is:
            // shapesA and shapesB are set to the current shapes in graphA_fig
            // and graphB_fig respectively. If graphA or graphB triggered the
            // callback and this data contains shapes, then we populate with the
            // new shapes. Otherwise the shapes from before are used.
            if ((triggered == "graph-A.relayoutData") && ("shapes" in graphA)) {
                shapesA = graphA.shapes;
            }
            if ((triggered == "graph-B.relayoutData") && ("shapes" in graphB)) {
                shapesB = graphB.shapes;
            }
            UndoState_track_changes(undo_data,[shapesA,shapesB]);
        }
    }
    new_graphA_fig.layout.shapes=shapesA;
    new_graphB_fig.layout.shapes=shapesB;
    console.log("new_graphA_fig",new_graphA_fig);
    console.log("new_graphB_fig",new_graphB_fig);
    return [undo_data,new_graphA_fig,new_graphB_fig];
}
""",
[Output("undo-data","data"),
 Output("graph-A","figure"),
 Output("graph-B","figure")],
[Input("graph-A","relayoutData"),
 Input("graph-B","relayoutData"),
 Input("undo-button","n_clicks"),
 Input("redo-button","n_clicks")],
[State("undo-data","data"),
 State("graph-A","figure"),
 State("graph-B","figure")])


if __name__ == '__main__':
    app.run_server(debug=True)

# Conclusion: callback keeps getting passed both figures containing shapes if
# shapes have been drawn.

