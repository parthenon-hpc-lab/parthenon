# Parthenon Python Metrics Github Application

This package is responsible for running the performance metrics of the pathernon project.

githubapp.py - contains a generic class for interacting with the github RESTful api

parthenon_metrics_app.py - is the parthenon specefic metrics app which is a child of the github app
base class

partheon_performance_json_parser.py - contains a class for parsing the unique structure of the 
parthenon metrics data which is output in json format

parthenon_performance_advection_analyzer - is a class that is responsible for analyzing the output
of the advection performance tests

parthenon_performance_plotter - provides plotting functionality
