function s2ab(s){
    var buf = new ArrayBuffer(s.length);
    // console.log("data string length" + s.length)
    var view = new Uint8Array(buf);
    for (var i=0; i!=s.length; ++i) view[i] = s.charCodeAt(i) & 0xFF;
    return buf;
}
var data=data
const filename = 'data_result.xlsx'
const blob = new Blob([s2ab(atob(data))], { type: ''})

//addresses IE
if (navigator.msSaveBlob) {
    navigator.msSaveBlob(blob, filename)
} else {
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = filename
    link.target = '_blank'
    link.style.visibility = 'hidden'
    link.dispatchEvent(new MouseEvent('click'))
}
