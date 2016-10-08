/**
* General page controls for demos and tabs...
*
* Copyright (C) 2014 Texas Instruments Incorporated - http://www.ti.com/
* 
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*  
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*  
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
* LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
* OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
* WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

$(document).ready(function(){ 

	var filename = window.location.href.substr(window.location.href.lastIndexOf("/")+1);
	if (filename.indexOf('overview') > -1) {
		$("#overview").toggleClass("active");
	} else if (filename.indexOf('about') > -1) {
		$("#about").toggleClass("active");
	} else if (filename.indexOf('setup') > -1) {
		$("#setup").toggleClass("active");
	} else if (filename.indexOf('portal') > -1) {
		$("#portal").toggleClass("active");
	} else if (filename.indexOf('mcudemo') > -1) {
		$("#mcudemo").toggleClass("active");
	}

});