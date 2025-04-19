// window.addEventListener("load", function(event)
// {
//     var images = document.getElementsByTagName("img");
//     for(var i=0; i < images.length; i++) {
//         // images[i].src = "someImage.jpg";
//         // console.log(images[i].src);
//     }
// });

// How to embed PDF files in sphinx using Adobe Embed PDF API:
//
// 1. In /docs/source/_templates/layout.html add this to the scripts:
// <script src="https://documentcloud.adobe.com/view-sdk/viewer.js"></script>
//
// 2. In the sphinx rts file, create a div by the following raw html directive.
//    Make sure the div id is the same.
//
// .. raw:: html
//
//     <div id="adobe-dc-view" style="width: 800px;"></div>
//
// 3. Then uncomment the following function here, and add the url of the PDF
//    file.

// My Adobe Embedded API user ID
const clientId = "77a1b26570924910a4a6e24350b76f43";

// =================================
// Options to show Text (non-slides)
// =================================

const viewerOptionsForText = {
    // embedMode: "IN_LINE",
    embedMode: "LIGHT_BOX",
    // embedMode: "SIZED_CONTAINER",
    // embedMode: "FULL_WINDOW",
    defaultViewMode: "FIT_PAGE",
    // showDownloadPDF: false,
    // showPrintPDF: false,
    enableFormFilling: false,
    // showZoomControl: false,
    showThumbnails: false,
    showBookmarks: false,
    showAnnotationTools: false,
    showFullScreen: true,
    // enableLinearization: true,
    showDownloadPDF: false,
    showPrintPDF: false,
    exitPDFViewerType: "CLOSE",
};

// ======================
// Options to show slides
// ======================

const viewerOptionsForSlide = {
    // embedMode: "IN_LINE",
    embedMode: "LIGHT_BOX",
    // embedMode: "SIZED_CONTAINER",
    // embedMode: "FULL_WINDOW",
    defaultViewMode: "FIT_PAGE",
    // defaultViewMode: "FIT_WIDTH",
    // showDownloadPDF: false,
    // showPrintPDF: false,
    enableFormFilling: false,
    showZoomControl: false,
    showThumbnails: false,
    showBookmarks: false,
    showAnnotationTools: false,
    showFullScreen: true,
    // enableLinearization: true,
    showDownloadPDF: false,
    showPrintPDF: false,
    exitPDFViewerType: "CLOSE",
};

// =========
// fetch PDF
// =========

function fetchPDF(urlToPDF) {
    return new Promise((resolve) => {
        fetch(urlToPDF)
            .then((resolve) => resolve.blob())
            .then((blob) => {
                resolve(blob.arrayBuffer());
            })
    })
}

// ========
// show PDF
// ========

function showPDF(urlToPDF, slide=false, allowTextSelection) {

    var adobeDCView = new AdobeDC.View({
            clientId: clientId
        });

    var viewerOptions = null;
    if (slide == true) {
        viewerOptions = viewerOptionsForSlide;
    } else {
        viewerOptions = viewerOptionsForText;
    }

    var previewFilePromise = adobeDCView.previewFile(
        {
            content: { promise: fetchPDF(urlToPDF) },
            metaData: { fileName: urlToPDF.split("/").slice(-1)[0] }
        },
        viewerOptions
    );

    // Allow text selection
    previewFilePromise.then(adobeViewer => {
        adobeViewer.getAPIs().then(apis => {
            apis.enableTextSelection(allowTextSelection)
                .then(() => console.log("Success"))
                .catch(error => console.log(error));
         });
    });

    // Zoom if not slide
    // var zoomLevel = 1.2;
    // if (slide == false) {
    //     previewFilePromise.then(adobeViewer => {
    //         adobeViewer.getAPIs().then(apis => {
    //                 apis.getZoomAPIs().setZoomLevel(zoomLevel)
    //                         .then(result => console.log(result))
    //                         .catch(error => console.log(error));
    //         });
    //     });
    // }
}

// =============================
// List of PDF files information
// =============================

const pdfData = [
    {
        id: ["showPaper01"],
        url: "https://arxiv.org/pdf/2412.18407",
        slide: false,
        allowTextSelection: true,
    },
    {
        id: ["showSlide01",],
        url: "https://www.dropbox.com/scl/fi/i3dil2qk3hqwkfrtsf2iy/A-Statistical-Framework-for-Ranking-LLM-Based-Chatbots.pdf?rlkey=a7z039wcukswk1pdcfhm4qlh6&st=qualuwyb&dl=0",
        // url: "https://leaderbot.org/slides.pdf",
        slide: true,
        allowTextSelection: false,
    },
]

// ========================
// Direct Link From Dropbox
// ========================

// Converts a standard Dropbox link to a direct download link.

function directLinkFromDropboxLink(dropboxLink) {
    return dropboxLink.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "");
}

// =======================
// Direct Link From Github
// =======================

// Converts a standard Github link to a direct download link
// This script converts:
//     "https://github.com/ameli/ameli.github.io/blob/main/assets/files/cv.pdf"
// to
//     "https://ameli.github.io/assets/files/cv.pdf"

function directLinkFromGithubLink(githubLink) {
    var reg = /github.com\/[\s\S]*?\//;
    url = githubLink.replace(reg, "").replace("blob/main/", "");
    return url;
}

// =========================================
// Add Adobe Embedded event for each element
// =========================================

document.addEventListener("adobe_dc_view_sdk.ready", function () {

    for (const data of pdfData) {
        for (const id of data.id){
            el = document.getElementById(id)

            if (el) {
                el.addEventListener("click", function () {

                    var url = data.url;

                    // If the url is a standard share link from dropbox,
                    // convert it to direct download link
                    if (url.includes("www.dropbox.com")) {
                        url = directLinkFromDropboxLink(url);
                    }

                    // If the url is a standard share link from dropbox,
                    // convert it to direct download link
                    if (url.includes("github.com")) {
                        url = directLinkFromGithubLink(url);
                    }

                    // Show pdf with Adobe Embed
                    showPDF(url, data.slide, data.allowTextSelection)
                });
            }
        }
    } 
});

// ========================================
// Add arrayBuffer if necessary i.e. Safari
// ========================================

(function () {
    if (Blob.arrayBuffer != "function") {
        Blob.prototype.arrayBuffer = myArrayBuffer;
    }

    function myArrayBuffer() {
        return new Promise((resolve) => {
            let fileReader = new FileReader();
            fileReader.onload = () => {
                resolve(fileReader.result);
            };
            fileReader.readAsArrayBuffer(this);
        });
    }
})();


// document.addEventListener("adobe_dc_view_sdk.ready", function()
// {
//     var adobeDCView = new AdobeDC.View({clientId: "becbabeb5d0d4204b5b99689751e71ef", divId: "adobe-dc-view"});
//     adobeDCView.previewFile(
//         {
//             content:{location: {url: "https://arxiv.org/pdf/2009.07385.pdf"}},
//             // content:{location: {promise: filePromise}},
//             // metaData:{fileName: "Bodea Brochure.pdf"}
//             metaData:{fileName: ""}
//         },
//         {
//             // embedMode: "IN_LINE",
//             // embedMode: "LIGHT_BOX",
//             // embedMode: "SIZED_CONTAINER",
//             embedMode: "FULL_WINDOW",
//             showDownloadPDF: false,
//             showPrintPDF: false,
//             enableFormFilling: false,
//             showZoomControl: false,
//             showThumbnails: false,
//             showBookmarks: false,
//             showAnnotationTools: false,
//             showFullScreen: true,
//             enableLinearization: true,
//         }
//     );
// });

// =====================================
// Change the link of logo to index.html
// =====================================

document.addEventListener("DOMContentLoaded", function() {

    // Find an "<a>" element whose href ends with "contents.html"
    const match = document.querySelector("a[href*='contents.html']");

    // Replace "contents.html" with "index.html"
    if (match){
        match.href = match.href.replace('contents.html', 'index.html')
    }
});
