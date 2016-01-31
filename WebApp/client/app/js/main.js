/*
 * Copyright (C) 2016 by Jeshua Kracht
 */

(function () {

  'use strict';

  var mainCtrl = require('./controllers/main-controller');

  angular.module('RApp', ['ngRoute', 'ui.bootstrap'])

  .config([
    '$locationProvider',
    '$routeProvider',
    function($locationProvider, $routeProvider) {
      $locationProvider.hashPrefix('!');
      // routes
      $routeProvider
        .when("/", {
          templateUrl: "./partials/stats.html",
          controller: "MainController"
        })
        .otherwise({
           redirectTo: '/'
        });
    }
  ])

  // Load controller
  .controller('MainController', ['$scope', '$http', '$location', mainCtrl]);
}());