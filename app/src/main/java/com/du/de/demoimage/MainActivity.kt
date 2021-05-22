package com.du.de.demoimage

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.camera.core.ImageCapture
import com.du.de.demoimage.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.util.concurrent.ExecutorService

@Suppress("DEPRECATION")
class MainActivity : AppCompatActivity() {

    private lateinit var bitmap: Bitmap
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.pvViewFinder)
        val fileName = "labels.txt"
        val inputString = application.assets.open(fileName).bufferedReader().use { it.readLine() }
        val townList = inputString.split("\n")

        val tvText = findViewById<TextView>(R.id.tvItemName)

        val select = findViewById<Button>(R.id.btnSelect)
        select.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            "image/".also { intent.type = it }

            startActivityForResult(intent, 100)

        }

        val resized: Bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val model = MobilenetV110224Quant.newInstance(this)

        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)

        val buffer = TensorImage.fromBitmap(resized)
        val byteBuffer = buffer.buffer
        inputFeature0.loadBuffer(byteBuffer)

        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val max = getMax(outputFeature0.floatArray)
        tvText.text = townList[max]
        model.close()

        val predict = findViewById<Button>(R.id.btnPredict)
        predict.setOnClickListener(View.OnClickListener {
            Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        imageView.setImageURI(data?.data)
        val uri: Uri? = data?.data
        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)

    }

    private fun getMax(arr: FloatArray): Int {
        var index = 0
        var min = 0.0f

        for (i in 0..100) {
            if (arr[i] > min) {
                index = 1
                min = arr[i]
            }
        }
        return index
    }
}