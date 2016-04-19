/*
 * Copyright (C) 2016 by Jeshua Kracht
 */

/*
 * JavaScript R Application
 *
 * AngularJS ng-controller
 *
 * Initialization:
 *
 * Establishes WS connection on /r, subscribes on /topic/r.
 *
 * Events:
 *
 * ROUTPUT - RTask result message.
 * RUNTIMESTATS - RBroker runtime statistics message.
 * CLIENTALERT - RBroker runtime (error) notification message.
 *
 * User driven (index.html) events:
 *
 * Resize click -> $scope.resizePool() -> POST:/r/pool/init/{size}
 *
 * Execute click -> $scope.executeTasks() -> GET:/r/output/{tasks}
 *
 */
module.exports = function($scope, $http, $location) {

   //
   // ng-controller model on $scope.
   //
   $scope.brokerInitialized = false;
   $scope.alertMessage = null;

   $scope.results = [];
   $scope.poolSize = 1;
   $scope.taskCount = 1;

   $scope.runtimeStats = {
      requestedPoolSize: 1,
      allocatedPoolSize: 1,
      submittedTasks: 0,
      successfulTasks: 0,
      failedTasks: 0,
      averageCodeExecution: 0,
      averageServerOverhead: 0,
      averageNetworkLatency: 0
   };

   $scope.targetTaskThroughput = 0;
   $scope.currentTaskThroughput = 0;
   $scope.startTaskThroughput = 0;
   $scope.secondTaskThroughput = 0;
   $scope.minuteTaskThroughput = 0;

   //
   // Resize Button Handler:
   //
   $scope.resizePool = function() {
      $scope.alertMessage = 'RBroker pool is initializing. ' +
         'Requested ' + $scope.poolSize + ' R session(s) in the pool. ' +
         'This may take some time. Please wait.';
      $scope.brokerInitialized = false;

      console.log('Attempt to resize pool succeeded, new size=' + $scope.poolSize);

      $http.post('/r/pool/init/' + $scope.poolSize)
         .success(function(data, status, headers, config) {
            console.log('Attempt to resize pool succeeded, new size=' + $scope.poolSize);
         }).error(function(data, status, headers, config) {
            $scope.errorMessage = 'Attempt to resize pool failed, error=' + data;
         }).finally(function() {
            $scope.results = [];
            $scope.brokerInitialized = true;
            $scope.currentTaskThroughput = 0;
            $scope.secondTaskThroughput = 0;
            $scope.minuteTaskThroughput = 0;
         });
   };

   //
   // Execute Button Handler:
   //
   $scope.executeTasks = function() {
      $scope.currentTaskThroughput = 0;
      $scope.secondTaskThroughput = 0;
      $scope.minuteTaskThroughput = 0;
      $scope.targetTaskThroughput = $scope.taskCount;
      $scope.startTaskThroughput = Date.now();

      $http.get('/r/output/' + $scope.taskCount)
         .success(function(data, status, headers, config) {
            console.log('Attempt to execute tasks succeeded, taskCount=' + $scope.taskCount);
         }).error(function(data, status, headers, config) {
            $scope.errorMessage = 'Can\'t retrieve output!';
            $scope.errorMessage = 'Attempt to execute tasks failed, error=' + data;
         });
   };

   var primus = Primus.connect('ws://localhost:' + $location.port());

   // Subscribe for events on /topic/r.
   primus.on('open', function() {

      primus.on('/topic/r', function(msgObj) {

         if (msgObj.msgType === 'ROUTPUT') {

            var elapsedTime = Date.now() - $scope.startTaskThroughput;

            // $apply to propgate change to model.
            $scope.$apply(function() {

               $scope.currentTaskThroughput += 1;
               var throughput =
                  (1000 / elapsedTime) * $scope.currentTaskThroughput;
               $scope.secondTaskThroughput =
                  +(Math.round((throughput - (throughput % 0.01)) + 'e+2') + 'e-2');
               $scope.minuteTaskThroughput =
                  Math.round($scope.secondTaskThroughput * 60);

               // Discard older results from results
               // list to prevent browser rendering exhaustion.
               if ($scope.results.length > 300) {
                  $scope.results.length = 150;
               }
               $scope.results.unshift(msgObj);
            });

         } else if (msgObj.msgType === 'RUNTIMESTATS') {
            // $apply to propogate change to model.
            $scope.$apply(function() {
               $scope.alertMessage = null;
               $scope.runtimeStats = msgObj;
            });
         } else if (msgObj.msgType === 'CLIENTALERT') {
            // $apply to propogate change to model.
            $scope.$apply(function() {
               $scope.alertMessage = msgObj.msg;
            });
         }
      });

      //
      // Initialize initial RBroker pool on application startup.
      //
      $scope.$apply(function() { $scope.resizePool(); });
   });
};