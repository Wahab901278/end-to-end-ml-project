from flask import Flask, render_template, request, jsonify
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Initialize prediction pipeline
try:
    pipeline = PredictPipeline()
except Exception as e:
    print(f"Warning: Could not initialize pipeline: {e}")
    pipeline = None


@app.route('/', methods=['GET'])
def home():
    """Home route with API documentation."""
    return jsonify({
        'message': 'Welcome to Math Score Prediction API',
        'version': '1.0',
        'endpoints': {
            'POST /predict': 'Make a prediction for math score',
            'POST /predict-batch': 'Make predictions for multiple students (coming soon)'
        },
        'example_request': {
            'gender': 'female',
            'race_ethnicity': 'group B',
            'parental_level_of_education': "bachelor's degree",
            'lunch': 'standard',
            'test_preparation_course': 'completed',
            'reading_score': 85,
            'writing_score': 88
        }
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict math score based on student features.

    Expected JSON format:
    {
        "gender": "female",
        "race_ethnicity": "group B",
        "parental_level_of_education": "bachelor's degree",
        "lunch": "standard",
        "test_preparation_course": "completed",
        "reading_score": 85,
        "writing_score": 88
    }
    """
    try:
        if pipeline is None:
            return jsonify({
                'success': False,
                'error': 'Prediction pipeline not initialized'
            }), 500

        # Get JSON data from request
        data = request.get_json()

        # Validate required fields
        required_fields = [
            'gender', 'race_ethnicity', 'parental_level_of_education',
            'lunch', 'test_preparation_course', 'reading_score', 'writing_score'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Create CustomData object
        custom_data = CustomData(
            gender=data['gender'],
            race_ethnicity=data['race_ethnicity'],
            parental_level_of_education=data['parental_level_of_education'],
            lunch=data['lunch'],
            test_preparation_course=data['test_preparation_course'],
            reading_score=int(data['reading_score']),
            writing_score=int(data['writing_score'])
        )

        # Make prediction
        prediction_result = pipeline.predict_with_confidence(custom_data)

        return jsonify({
            'success': True,
            'data': {
                'student_features': {
                    'gender': data['gender'],
                    'race_ethnicity': data['race_ethnicity'],
                    'parental_level_of_education': data['parental_level_of_education'],
                    'lunch': data['lunch'],
                    'test_preparation_course': data['test_preparation_course'],
                    'reading_score': int(data['reading_score']),
                    'writing_score': int(data['writing_score'])
                },
                'prediction': {
                    'predicted_math_score': round(prediction_result['predicted_math_score'], 2),
                    'model_type': prediction_result['model_type']
                }
            }
        }), 200

    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input: {str(e)}'
        }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    pipeline_status = 'ready' if pipeline is not None else 'not initialized'
    return jsonify({
        'status': 'healthy',
        'pipeline': pipeline_status
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'message': 'Use GET / for API documentation'
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting Math Score Prediction API")
    print("="*80)
    print("\nAPI Documentation:")
    print("  Home: http://localhost:8000/")
    print("  Predict: POST http://localhost:8000/predict")
    print("  Health: http://localhost:8000/health")
    print("\n" + "="*80 + "\n")

    app.run(debug=True, host='0.0.0.0', port=4000)
