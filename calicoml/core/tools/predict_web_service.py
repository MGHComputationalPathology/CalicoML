# -*- coding: utf-8 -*-

"""\
Copyright (c) 2015-2018, MGH Computational Pathology

"""

from argparse import ArgumentParser
import json
import numpy as np
import pandas as pd
from calicoml.core.serialization.model import ClassificationModel

from flask import Flask, send_from_directory, request
import pkg_resources

app = Flask(__name__, static_url_path=pkg_resources.resource_filename('calicoml.core.tools', 'predict_ui'))
model = None


def format_datatypes(datatypes, whitelist=None):
    """Interprets and formats numpy/python datatypes as either 'numeric' or 'text'"""
    def format_one(type_obj):
        """Formats a single datatype"""
        type_obj = str(type_obj)
        if 'float' in type_obj or 'int' in type_obj:
            return 'numeric'
        elif 'object' in type_obj or 'unicode' in type_obj or 'str' in type_obj:
            return 'text'
        elif 'bytes' in type_obj:
            return 'binary'
        else:
            raise ValueError("Unknown data type: {}".format(str(type_obj)))

    if whitelist is None:
        whitelist = list(datatypes.keys())
    return {feat: format_one(dt) for feat, dt in datatypes.items() if feat in whitelist}


@app.route('/info')
def get_model_info():
    """Returns model metadata as a JSON string"""
    return json.dumps({'name': str(model.approach),
                       'features': model.features,
                       'datatypes': format_datatypes(model.training_problem.datatypes, model.features),
                       'outcome': model.outcome,
                       'training_set': {'path': model.training_problem.data.info.get('path', None),
                                        'auc': model.training_auc,
                                        'n_samples': model.training_problem.n_samples,
                                        'n_features': model.training_problem.n_features,
                                        'prevalence': model.training_problem.prevalence},
                       'positive_outcome': model.positive_outcome})


@app.route('/training_data')
def get_sample_data(df=None):
    """Converts a data frame with sample data into a JSON array. Returns a JSON string."""
    def sanitize_nan(val):
        """\
        Pandas represents empty fields with NaNs, which JSON does not support. Here we
        explicitly replace them with empty strings.

        """
        return "" if isinstance(val, (float, np.float)) and np.isnan(val) else val

    df = df if df is not None else model.training_problem.dataframe
    features_of_interest = model.training_problem.features + [model.outcome]
    return json.dumps([{feat: sanitize_nan(row[feat]) for feat in features_of_interest}
                       for _, row in df.iterrows()])


@app.route('/predict', methods=["POST"])
def predict():
    """Runs prediction"""
    def get_one_prediction(idx, row):
        """Gets a prediction object for a single sample row"""
        return {'sample_id': idx, 'score': row['score']}

    input_df = pd.DataFrame(request.get_json())
    predictions_df = model.predict(input_df)
    return json.dumps({'scores': [get_one_prediction(idx, row) for idx, row in predictions_df.iterrows()]})


@app.route('/ui/<path:path>')
def serve_static_file(path):
    """Serves static resources"""
    return send_from_directory(app.static_url_path, path)


@app.route('/upload', methods=["POST"])
def receive_upload():
    """Handles file upload"""
    df = pd.read_csv(request.files['file'].stream, sep='\t')
    return get_sample_data(df)


def main(args=None):
    """The main method"""
    global model  # pylint: disable=global-statement

    parser = ArgumentParser()
    parser.add_argument('model', help="Model to use")
    parser.add_argument('--port', type=int, default=5100, help="Port on which to run the service")
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode")
    parsed_args = parser.parse_args(args)

    model = ClassificationModel.read(parsed_args.model)
    app.run(port=parsed_args.port, debug=parsed_args.debug)


if __name__ == "__main__":
    main()
