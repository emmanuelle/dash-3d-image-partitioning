function* _range_gen(N) {
    let n=N; while(n>0){yield --n;}
}

// We cannot use this syntax because we want UndoState to be JSON serializable.
// All the methods below will work on objects with the following fields.
//function UndoState(n_shape_lists) {
//    self.undo_n_clicks=0;
//    self.redo_n_clicks=0;
//    self.undo_shapes=[];
//    self.redo_shapes=[];
//    self.n_shape_lists=n_shape_lists;
//}

// Return true if shapes are different from the last shapes stored in
// undo_shapes
UndoState_shapes_changed = function (self,shapes) {
    return !(shapes == self.undo_shapes[-1]);
}

// Add new shapes and empty the redo list (because if redos could be called,
// they are now irrelevant after adding a new shape)
UndoState_add_new_shapes = function (self,shapes) {
    self.redo_shapes = [];
    self.undo_shapes.append(shapes);
}

// Add new shapes to the undo list if shapes are new
UndoState_track_changes = function (self,shapes) {
    if (UndoState_shapes_changed(self,shapes)) {
        UndoState_add_new_shapes(self,shapes);
    }
}

UndoState__return_last_undo = function (self) {
    if (!self.undo_shapes.length) {
        return Array.from(
            _range_gen(self.n_shape_lists),
            ()=>[]);
    }
    let ret = self.undo_shapes[-1];
    return ret;
}    

// Undo returns previous shapes and stores the current shapes in the redo list
UndoState_apply_undo = function (self) {
    if (self.undo_shapes.length) {
        self.redo_shapes.append(self.undo_shapes.pop());
    }
    return UndoState__return_last_undo(self);
}

// Redo puts last redo on top of the undo list and returns the set of shapes at
// the end of the undo list
UndoState_apply_redo = function (self) {
    if (self.redo_shapes.length) {
        self.undo_shapes.append(self.redo_shapes.pop());
    }
    return UndoState__return_last_undo(self);
}
