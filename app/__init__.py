from flask import Flask, jsonify
from flask_cors import CORS

from app.core.config import Config
from app.routes.simulation_routes import simulation_bp
from app.routes.scheduler_routes import scheduler_bp
from app.routes.forecasting_routes import forecasting_bp
from app.routes.datasets_routes import datasets_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Enable CORS for React frontend (http://localhost:5173)
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Register API Blueprints
    app.register_blueprint(simulation_bp, url_prefix=f"{Config.API_PREFIX}/simulation")
    app.register_blueprint(scheduler_bp, url_prefix=f"{Config.API_PREFIX}/scheduler")
    app.register_blueprint(forecasting_bp, url_prefix=f"{Config.API_PREFIX}/forecasting")
    app.register_blueprint(datasets_bp, url_prefix=f"{Config.API_PREFIX}/datasets")
    
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            "status": "healthy",
            "framework": "Flask 3.x",
            "service": Config.PROJECT_NAME,
            "version": Config.VERSION
        }), 200

    @app.route('/', methods=['GET'])
    def index():
        return jsonify({
            "message": "Welcome to Cloud RL Task Scheduler & Forecaster Backend API (Flask)",
            "health": "/health",
            "api_v1": Config.API_PREFIX
        }), 200
        
    return app
