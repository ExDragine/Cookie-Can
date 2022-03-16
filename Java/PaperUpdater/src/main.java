import org.json.JSONArray;
import org.json.JSONObject;

import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Path;
import java.nio.file.Paths;

public class main {
    public static void main(String args[]) throws IOException {
        JSONObject jsonObj1 = GetJSONObjectByUrl("http://launchermeta.mojang.com/mc/game/version_manifest.json");
        JSONObject latest = jsonObj1.getJSONObject("latest");
        String MinecraftVersion = latest.getString("release");
        MinecraftVersion = MinecraftVersion.substring(0, 4);
        System.out.println(MinecraftVersion);

        String PaperURL = "https://papermc.io/api/v2";

        JSONObject VersionGroupJson = GetJSONObjectByUrl(
                PaperURL + "/projects/paper/version_group/" + MinecraftVersion + "/builds");
        JSONArray VersionsList = VersionGroupJson.getJSONArray("versions");
        // String version =
        // VersionsList.getJSONObject(VersionsList.length()-1).toString();
        String version = VersionsList.get(VersionsList.length() - 1).toString();
        System.out.println(version);

        JSONArray BuildsList = VersionGroupJson.getJSONArray("builds");
        JSONObject BuildObject = BuildsList.getJSONObject(BuildsList.length() - 1);
        int build = BuildObject.getInt("build");
        System.out.println(build);

        String download = BuildObject.getJSONObject("downloads").getJSONObject("application").getString("name");
        System.out.println(download);
        String DownloadUrl = PaperURL + "/projects/paper/versions/" + version + "/builds/" + build + "/downloads/"
                + download;
        Path path = Paths.get("");
        String currentDir = path.toAbsolutePath().toString();

        // String SavePath = currentDir + "\\" + MinecraftVersion + '-' + build +
        // ".jar";
        String SavePath = currentDir + "\\" + "paper" + ".jar";
        downloadA(DownloadUrl, SavePath);
        System.out.println("download done");
        // tmp =
        // requests.api.get(PaperURL+'/projects'+''+str(version)+'/builds/'+str(build)+"downloads/"+str(download))
        // build = VersionGroup['builds'][-1]['build']
    }

    /**
     * 根据链接地址下载文件
     * 
     * @param downloadUrl 文件链接地址
     * @param path        下载存放文件地址 + 文件名
     */
    private static void downloadA(String downloadUrl, String path) {
        URL url = null;
        DataInputStream dataInputStream = null;
        FileOutputStream fileOutputStream = null;
        try {
            url = new URL(downloadUrl);
            dataInputStream = new DataInputStream(url.openStream());
            fileOutputStream = new FileOutputStream(new File(path));
            ByteArrayOutputStream output = new ByteArrayOutputStream();

            byte[] buffer = new byte[1024];
            int length;

            while ((length = dataInputStream.read(buffer)) > 0) {
                output.write(buffer, 0, length);
            }
            fileOutputStream.write(output.toByteArray());
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (dataInputStream != null) {
                try {
                    dataInputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            if (fileOutputStream != null) {
                try {
                    fileOutputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    public static JSONObject GetJSONObjectByUrl(String url) throws IOException {
        URL url1 = new URL(url);
        BufferedReader reader = null;
        reader = new BufferedReader(new InputStreamReader(url1.openStream()));

        String s;
        String Json1 = "";
        while ((s = reader.readLine()) != null) {
            // System.out.println(s);
            Json1 += s;
        }
        reader.close();
        JSONObject jsonObj1 = new JSONObject(Json1);
        return jsonObj1;
    }
}
