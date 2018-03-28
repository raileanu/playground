import abc
import logging
import json
from flask import Flask, jsonify, request

logger = logging.getLogger(__name__)


class DockerAgentRunner(metaclass=abc.ABCMeta):

    """Abstract base class to implement Docker-based agent"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def act(self, observation, action_space):
        """Given an observation, returns the action the agent should"""
        raise NotImplementedError()

    def run(self, host="0.0.0.0", port=10080):
        """Runs the agent by creating a webserver that handles action requests."""
        app = Flask(self.__class__.__name__)

        @app.route("/action", methods=["POST"])
        def action(): #pylint: disable=W0612
            data = request.get_json()
            observation = data.get("obs")
            observation = json.loads(observation)
            action_space = data.get("action_space")
            action_space = json.loads(action_space)
            action = self.act(observation, action_space)
            return jsonify({"action": action})

        @app.route("/ping", methods=["GET"])
        def ping(): #pylint: disable=W0612
            return jsonify(success=True)

        logger.info("Starting agent server on port %d", port)
        app.run(host=host, port=port)