<!-- Copyright (c) 2013-2016 Texas Instruments Incorporated.  All rights reserved. -->
var HTTPrequest = new XMLHttpRequest();

window.onload = function()
{
    document.getElementById('about').onclick = loadAbout;
    document.getElementById('overview').onclick = loadOverview;
    document.getElementById('block').onclick = loadBlock;
    document.getElementById('io').onclick = loadIO;
    loadPage("about.htm");
}

function loadAbout()
{
    loadPage("about.htm");
    return false;
}

function loadOverview()
{
    loadPage("overview.htm");
    return false;
}

function loadBlock()
{
    loadPage("block.htm");
    return false;
}

function loadIO()
{
    loadPage("io.htm");
    ledstateGet();
    speedGet();
    return false;
}

function toggle_led()
{
    var params = "__SL_P_U00";
    params = params +"LED_TOGGLE";

    HTTPrequest.open("POST","No_content", true);
    HTTPrequest.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    HTTPrequest.onreadystatechange = function()
    {
        if (HTTPrequest.readyState==4 && HTTPrequest.status==204)
        {
            ledstateGet();
        }
    }
    HTTPrequest.send(params);
}

function speedSet()
{
    var params = "__SL_P_U01";
    var speed_txt = document.getElementById("speed_percent").value;

    params = params+ "_" + speed_txt + "END";
    HTTPrequest.open("POST","No_content", true);
    HTTPrequest.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    HTTPrequest.onreadystatechange = function()
    {
        if (HTTPrequest.readyState==4 && HTTPrequest.status==204)
        {
            speedGet();
        }
    }
    HTTPrequest.send(params);
}

function loadPage(page)
{
    if(window.XMLHttpRequest)
    {
        xmlhttp = new XMLHttpRequest();
    }
    else
    {
        xmlhttp = new ActiveXObject("Microsoft.XMLHTTP");
    }

    xmlhttp.open("GET", page, true);
    xmlhttp.setRequestHeader("Content-type",
                             "application/x-www-form-urlencoded");
    xmlhttp.send();

    xmlhttp.onreadystatechange = function ()
    {
        if((xmlhttp.readyState == 4) && (xmlhttp.status == 200))
        {
            document.getElementById("content").innerHTML = xmlhttp.responseText;
        }
    }
}
