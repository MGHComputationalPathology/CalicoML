/*

   UI for displaying/inputting sample data

   Copyright (c) 2015-2018, MGH Computational Pathology
   All rights reserved

*/


/* Stop JSHint from complaining about features from ECMAScript 6, like 'for of' */
/* jshint esnext: true */


/** Handles the grid for sample input */
var SampleGrid = function(target_elt, model, onCellChange, fetchPredictions) {
    this.target_elt = target_elt;
    this.model = model;
    this.sample_data = [];
    this.grid = null;
    this.onCellChange = onCellChange;
    this.fetchPredictions = fetchPredictions;
    this.invalidate();
};


/** Sets sample data displayed by the grid */
SampleGrid.prototype.setSamples = function(new_data) {
    this.sample_data.length = 0;
    for(var item of new_data) {
        this.sample_data.push(item);
    }
    this.grid.invalidate();
    this.fetchPredictions();
};


/** Redraws the grid. Use if either the model or sample data change */
SampleGrid.prototype.invalidate = function() {
    /** SlickGrid options */
    var options = {
        editable: true,
        enableAddRow: true,
        enableCellNavigation: true,
        asyncEditorLoading: false,
        autoEdit: false,
    };

    /** Init columns */
    columns = [];
    for(var feat of this.model.features) {
        columns.push({id: feat,
                      name: feat,
                      field: feat,
                      width: 200,
                      formatter: this.model.datatypes[feat] == "numeric" ? Slick.Formatters.FixedPrecisionFloatFormatter : null,
                      editor: this.model.datatypes[feat] == "numeric" ? Slick.Editors.Float : Slick.Editors.Text});
    }

    columns.push({id: this.model.outcome, name: this.model.outcome, field: this.model.outcome, width: 150});
    columns.push({id: 'predicted_score', name: 'Predicted Score', field: 'predicted_score', width: 150});
    columns.push({id: 'predicted_score_prc', name: '', field: 'predicted_score_prc',
                  formatter: Slick.Formatters.PercentCompleteBar, width: 200});

    for(var col of columns) {
        col.sortable = true;
        col.resizable = true;
    }

    /** Create SlickGrid */
    this.grid = new Slick.Grid(this.target_elt, this.sample_data, columns, options);
    this.grid.setSelectionModel(new Slick.CellSelectionModel());

    /* Add new item event */
    grid_instance = this;
    this.grid.onAddNewRow.subscribe(function(e, args) {
        grid_instance.grid.invalidateRow(grid_instance.sample_data.length);
        grid_instance.sample_data.push(args.item);
        grid_instance.grid.updateRowCount();
        grid_instance.grid.render();
    });

    /* Cell value changed */
    this.grid.onCellChange.subscribe(this.onCellChange);

    /* Sort column */
    this.grid.onSort.subscribe(function(e, args){
        var field = args.sortCol.field;

        grid_instance.sample_data.sort(function(a, b){
            var result =
                a[field] > b[field] ? 1 :
                a[field] < b[field] ? -1 :
                0;

            return args.sortAsc ? result : -result;
        });

        grid_instance.grid.invalidate();
    });
};
