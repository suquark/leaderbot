// My Adobe Embedded API user ID
const clientId = "becbabeb5d0d4204b5b99689751e71ef";

// ======================
// Options to show slides
// ======================

const slide_viewerOptionsForSlide = {
    // embedMode: "IN_LINE",
    // embedMode: "LIGHT_BOX",
    embedMode: "SIZED_CONTAINER",
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

function slide_fetchPDF(urlToPDF) {
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

function slide_showPDF(urlToPDF, slide=false, allowTextSelection) {

    var adobeDCView = new AdobeDC.View({
            clientId: clientId
        });

    var viewerOptions = null;
    if (slide == true) {
        viewerOptions = slide_viewerOptionsForSlide;
    } else {
        viewerOptions = viewerOptionsForText;
    }

    var previewFilePromise = adobeDCView.previewFile(
        {
            content: { promise: slide_fetchPDF(urlToPDF) },
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

// ========================
// Direct Link From Dropbox
// ========================

// Converts a standard Dropbox link to a direct download link.

function slide_directLinkFromDropboxLink(dropboxLink) {
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

function slide_directLinkFromGithubLink(githubLink) {
    var reg = /github.com\/[\s\S]*?\//;
    url = githubLink.replace(reg, "").replace("blob/main/", "");
    return url;
}

// ====
// Show
// ====

document.addEventListener("adobe_dc_view_sdk.ready", function () {

    // Get div element
    var id = "adobe-dc-view";
    el = document.getElementById(id)

    // Get url
    var url = el.dataset.url;

    // If the url is a standard share link from dropbox, convert it to direct download link
    if (url.includes("www.dropbox.com")) {
        url = slide_directLinkFromDropboxLink(url);
    }

    // If the url is a standard share link from dropbox, convert it to direct download link
    if (url.includes("github.com")) {
        url = slide_directLinkFromGithubLink(url);
    }

    // Show pdf with Adobe Embed
    slide = true
    allowTextSelection = false;
    slide_showPDF(url, slide, allowTextSelection)
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
