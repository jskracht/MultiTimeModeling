/*
 * Copyright (C) 2010-2015 by Revolution Analytics Inc.
 *
 * This program is licensed to you under the terms of Version 2.0 of the
 * Apache License. This program is distributed WITHOUT
 * ANY EXPRESS OR IMPLIED WARRANTY, INCLUDING THOSE OF NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE. Please refer to the
 * Apache License 2.0 (http://www.apache.org/licenses/LICENSE-2.0) for more 
 * details.
 */

'use strict';

var rbroker = require('rbroker'),
    print = require('../util/rbroker-print-helper'),
    config = require('../../config/config');

var RTopic = '/topic/r',
    MSG_TYPES = {
        runtime: 'RUNTIMESTATS',
        output: 'ROUTPUT',
        error: 'CLIENTALERT'
    };

var round = function (num) {
    return +(Math.round(num + 'e+2') + 'e-2');
};

function RService(primus) {
    this.primus = primus;
    this.broker = null;
    this.lastAllocatedPoolSize = 0;
    this.brokerConfig = {
        maxConcurrentTaskLimit: 0,
        host: process.env.endpoint || config.endpoint,
        credentials: {
            username: process.env.username || config.credentials.username,
            password: process.env.password || config.credentials.password
        },
        releaseGridResources: true,
        logging: config.logging,
        pool: {
            preloadobjectauthor: process.env.username || config.credentials.username,
            preloadobjectdirectory: config.constants.REPO_DIRECTORY
        }
    };
}

RService.prototype = {

    buildPool: function (poolSize) {
        var self = this;
        this.brokerConfig.maxConcurrentTaskLimit = poolSize;

        if (!this.broker) {
            this.attachBroker();
        } else {
            this.broker.shutdown()
                .then(function () {
                    console.log('Pooled: RBroker shutdown `successful`.');
                    self.attachBroker();
                }, function () {
                    console.log('Pooled: RBroker has shutdown `failure`.');
                });
        }
    },

    buildTask: function (task) {
        return rbroker.pooledTask({
            filename: config.constants.REPO_SCRIPT,
            directory: config.constants.REPO_DIRECTORY,
            author: this.brokerConfig.credentials.username,
            routputs: ['x']
        });
    },

    submit: function (task) {
        if (this.broker && task) {
            this.broker.submit(task);
        }
    },

    destroy: function () {
        if (this.broker) {
            this.broker.shutdown()
                .then(function () {
                    console.log('Pooled: RBroker shutdown `successful`.');
                    self.attachBroker();
                }, function () {
                    console.log('Pooled: RBroker has shutdown `failure`.');
                });
        }
    },

    /**
     * Push RuntimeStats message over STOMP Web Socket to clients
     * listening on RTopic.
     *
     * @api private
     */
    broadcast: function (message) {
        this.primus.send(RTopic, message);
    },

    /**
     * Attach and listen on a new PooledTaskBroker.
     * @api private
     */
    attachBroker: function () {
        var self = this;

        this.broker = rbroker.pooledTaskBroker(this.brokerConfig)
            .ready(function () {
                self.lastAllocatedPoolSize = self.broker.maxConcurrency();

                console.log('RBroker pool initialized with ' +
                    self.lastAllocatedPoolSize + ' R sessions.');
                self.broadcast(self.runtimeStats());
            })
            .complete(function (rTask, rTaskResult) {
                print.results(rTask, rTaskResult);

                // -- notify successful result --
                self.broadcast(self.buildROutput(rTask, rTaskResult));
            })
            .error(function (err) {
                print.error(err);

                // -- notify error --
                self.broadcast({
                    msgType: MSG_TYPES.error,
                    cause: err,
                    msg: 'The RBroker runtime has indicated an unexpected ' +
                    ' runtime error has occured. Cause: ' + err
                });
            })
            .progress(function (stats) {
                print.stats(stats);
                self.broadcast(self.runtimeStats(stats));
            });
    },

    /**
     * Private helper methods.
     * @api private
     */
    runtimeStats: function (stats) {

        var runtime = {
            msgType: MSG_TYPES.runtime,
            requestedPoolSize: this.brokerConfig.maxConcurrentTaskLimit,
            allocatedPoolSize: this.lastAllocatedPoolSize,
            endpoint: this.brokerConfig.host
        };

        if (this.brokerConfig.credentials) {
            runtime.username = this.brokerConfig.credentials.username;
        }

        if (stats) {
            runtime.submittedTasks = stats.totalTasksRun;
            runtime.successfulTasks = stats.totalTasksRunToSuccess;
            runtime.failedTasks = stats.totalTasksRunToFailure;

            runtime.averageCodeExecution = 0;
            runtime.averageServerOverhead = 0;
            runtime.averageNetworkLatency = 0;

            if (stats.totalTasksRunToSuccess > 0) {
                runtime.averageCodeExecution =
                    round(stats.totalTimeTasksOnCode / stats.totalTasksRunToSuccess);

                var avgTimeOnServer =
                    stats.totalTimeTasksOnServer / stats.totalTasksRunToSuccess;

                runtime.averageServerOverhead =
                    round(avgTimeOnServer - runtime.averageCodeExecution);

                var avgTimeOnCall =
                    stats.totalTimeTasksOnCall / stats.totalTasksRunToSuccess;

                runtime.averageNetworkLatency =
                    round(avgTimeOnCall - avgTimeOnServer);
            }
        }

        return runtime;
    },

    /**
     * Private helper methods.
     * @api private
     */
    buildROutput: function (rTask, rTaskResult) {
        return {
            msgType: MSG_TYPES.output,
            success: rTaskResult ? true : false
        };
    }

};

module.exports = RService;