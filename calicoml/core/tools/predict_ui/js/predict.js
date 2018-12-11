/*

   Experimental UI for prediction

   Copyright (c) 2015-2018, MGH Computational Pathology
   All rights reserved

*/


/* Stop JSHint from complaining about features from ECMAScript 6, like 'for of' */
/* jshint esnext: true */


/** SampleGrid instance for displaying samples and features */
var grid;


/** Asynchronously loads model metadata from the server */
loadModelInfo = function() {
    $.ajax({
        url: "../info",
        type: "GET",
        dataType: "json",
        contentType: "application/json",
    }).done(function(model) {
        updateModelInfo(model);
        grid = new SampleGrid("#data_container", model,
                              onCellChange = function (e, args) { fetchPredictions(); },
                              fetchPredictions = fetchPredictions);

    });
};


/** Loads the training data from server */
loadTrainingData = function() {
    $.ajax({
        url: "../training_data",
        type: "GET",
        dataType: "json",
        contentType: "application/json",
    }).done(function(data) {
        grid.setSamples(data);
    });
};


/** Updates the displayed model information */
updateModelInfo = function(model) {
    $("#model_name").html(model.name);
    $("#model_outcome").html(model.outcome + " == " + model.positive_outcome);
    $("#model_training_set").html(model.training_set.n_samples + " samples, " +
                                  model.training_set.n_features + " features, " +
                                  (model.training_set.prevalence * 100.0).toFixed(0) + "% prevalence" +
                                  " (" + model.training_set.path + ")");
    $("#model_training_auc").html(model.training_set.auc.toFixed(2));
};

/** Asynchronously gets predictions using current sample data */
fetchPredictions = function() {
    $.ajax({
        url: "../predict",
        data: JSON.stringify(grid.sample_data),
        dataType: "json",
        contentType: "application/json",
        type: "POST"
    }).done(function(predictions_obj) {
        var predictions = predictions_obj.scores;
        if(predictions.length != grid.sample_data.length) {
            alert('Error: Number of predictions does not match the number of samples.');
            return;
        }
        for(var i in predictions) {
            grid.sample_data[i].predicted_score = predictions[i].score.toFixed(3);
            grid.sample_data[i].predicted_score_prc = 100.0 * predictions[i].score;
        }
        grid.invalidate();
    });
};


/** onload hook */
window.onload = function() {
    loadModelInfo();

    $('#fileupload').fileupload({
        dataType: 'json',
        done: function (e, data) {
            grid.setSamples(data.result);
        }
    });
};
