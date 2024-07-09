function s2ab(s){
    var buf = new ArrayBuffer(s.length);
    var view = new Uint8Array(buf);
    for (var i=0; i!=s.length; ++i) view[i] = s.charCodeAt(i) & 0xFF;
    return buf;
}

const filename = 'data_result.xlsx'
const blob = new Blob([s2ab(atob(data))], { type: ''})