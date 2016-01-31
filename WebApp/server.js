/*
 * Copyright (C) 2016 by Jeshua Kracht
 */

var express      = require('express'),
    Primus       = require('primus.io'),
    config       = require('./config/config'), 
    RService     = require('./server/service/rservice'),
    app          = express(),    
    router       = express.Router();

app.use('/', router);
app.use(express.static(__dirname + '/client/app'));

// -- Start Primus server --
var server = require('http').createServer(app);
var primus = new Primus(server, { transformer: 'websockets', parser: 'JSON' });

primus.on('connection', function (spark) {
    var rService = new RService(primus);
   
    router.get('/r/output/:tasks', function(req, res) {
       var tasks = req.params.tasks === 0 ? 1 : req.params.tasks;
       console.log('REST:/r/output/' + tasks + ' called.');

       for(var i = 0; i < tasks; i++) {
          rService.submit(rService.buildTask());
       }
       
       res.json({ success: true });
    });

    router.post('/r/pool/init/:size', function (req, res) {
      var size = req.params.size === 0 ? 1 : req.params.size;
      console.log('REST:/pool/init/' + size + ' called.');

      rService.buildPool(size);
      res.json({ success: true });
    });
});

primus.on('disconnection', function () {
  console.log('disconnect...');
});

// -- Start server --
server.listen(config.port, function() {
  var endpoint = process.env.endpoint || config.endpoint,
      username = process.env.username || config.credentials.username;

  console.log('\n\n');
  console.log('==============================================================');
  console.log(' Project property (DeployR endpoint): ' + endpoint);
  console.log(' Project property (DeployR username): ' + username);
  console.log(' Project property (DeployR password): [HIDDEN] \n');
  console.log('\033[96m Example listening on http://localhost:' + config.port +' \033[39m');
  console.log('==============================================================');
});